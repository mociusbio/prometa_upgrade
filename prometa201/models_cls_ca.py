# -*- coding: utf-8 -*-
"""机制二 CLSCA: 双分支 Cross-Attention + token 池化拼接（预测无泄漏版）。

与 CLSDual 的区别: 融合不是发生在最后的 CLS 拼接, 而是用一个共享权重的
Cross-Attention 让两个组学的 token 互相"看"对方, 再池化成向量进下游。
关键设计是"两条支路、共享权重":

  表征支路 (双边信息, 进下游):
    toks_p = PathwayGatedTokenizer(x_prot)     # [B,359,64]
    toks_m = SimpleMetaEncoder(x_meta)         # [B,251,64]
    M_repr = CA(Q=toks_m, KV=toks_p)           # 代谢 token 被蛋白"告知"
    P_repr = CA(Q=toks_p, KV=toks_m)           # 蛋白 token 被代谢"告知"
    z_m = maxpool(M_repr); z_p = maxpool(P_repr)

  预测支路 (单边信息, 训完丢弃) —— 保证预训练时不泄漏目标模态:
    p2m: q = 可学习 query [251 个代谢插槽], KV = toks_p(只有蛋白)
         -> CA -> head_p2m -> 预测 x_meta [B,251]
    m2p: q = 可学习 query [359 个蛋白插槽], KV = toks_m(只有代谢)
         -> CA -> head_m2p -> 预测 x_prot [B,2923]
    (CA 层与表征支路共享权重; 两套 query 和头在 Stage 1 结束后丢弃)

  训练/预测信息不对称为类比: 像 BERT 训练时 mask、使用时完整输入 ——
  预训练只给单边信息逼模型学跨组学映射, 下游预测时两边都有。

Stage 2 (CLSDualCAMamlModel): 只用表征支路
    z = concat( z_p + shortcut_proj(x_prot),  z_m )  ->  Linear(128->1)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from prometa201.omics_encoders import PathwayGatedTokenizer, SimpleMetaEncoder
from prometa201.models_cls_dual import PerTokenPredictHead, DirectPerTokenHead


class CABlock(nn.Module):
    """Functional Cross-Attention Block (风格同 MetaTransformerLayer)。

    forward(query, kv): 自注意力式 CA —— q/k/v 分别投影, Q 来自 query, K/V 来自 kv,
    残差+LayerNorm + FFN(GELU)。支持 functional_forward 供 MAML inner loop 传参。
    """

    def __init__(self, embed_dim, num_heads, dropout=0.4, ffn_mult=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.linear1 = nn.Linear(embed_dim, embed_dim * ffn_mult)
        self.linear2 = nn.Linear(embed_dim * ffn_mult, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def functional_forward(self, query, kv, params, prefix):
        B, Lq, D = query.shape
        Lk = kv.shape[1]

        q = F.linear(query, params[f'{prefix}.q_proj.weight'], params[f'{prefix}.q_proj.bias'])
        k = F.linear(kv, params[f'{prefix}.k_proj.weight'], params[f'{prefix}.k_proj.bias'])
        v = F.linear(kv, params[f'{prefix}.v_proj.weight'], params[f'{prefix}.v_proj.bias'])

        q = q.view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, Lq, D)
        out = F.linear(out, params[f'{prefix}.out_proj.weight'], params[f'{prefix}.out_proj.bias'])

        x = F.layer_norm(query + F.dropout(out, p=self.dropout, training=self.training),
                         (D,), params[f'{prefix}.norm1.weight'], params[f'{prefix}.norm1.bias'])

        ff = F.linear(x, params[f'{prefix}.linear1.weight'], params[f'{prefix}.linear1.bias'])
        ff = F.gelu(ff)
        ff = F.dropout(ff, p=self.dropout, training=self.training)
        ff = F.linear(ff, params[f'{prefix}.linear2.weight'], params[f'{prefix}.linear2.bias'])
        ff = F.dropout(ff, p=self.dropout, training=self.training)

        x = F.layer_norm(x + ff, (D,), params[f'{prefix}.norm2.weight'], params[f'{prefix}.norm2.bias'])
        return x


def mlp_head(in_dim, out_dim, hidden=128, dropout=0.5):
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
        nn.Dropout(dropout), nn.Linear(hidden, out_dim))


def pool_tokens(t, mode='max'):
    return t.max(dim=1).values if mode == 'max' else t.mean(dim=1)


class PretrainCLSDualCA(nn.Module):
    """Stage 1: CA 版相互预测 + 自重建预训练（表征支路双边 / 预测支路单边，共享 ca_layers）"""

    def __init__(self, meta_dim=251, prot_dim=2923, d_model=64, nhead=2,
                 num_ca_layers=2, dropout=0.5, pathway_mask=None, unknown_indices=None,
                 pool='max'):
        super().__init__()
        self.pool = pool
        self.dropout_rate = dropout
        # unk token 数 = PathwayGatedTokenizer.hidden_mult(=32), 与 unknown_indices 长度无关
        num_prot_tokens = pathway_mask.size(0) + (32 if unknown_indices else 0)

        self.tokenizer = PathwayGatedTokenizer(prot_dim, d_model, pathway_mask, unknown_indices)
        self.meta_encoder = SimpleMetaEncoder(meta_dim, d_model)
        self.ca_layers = nn.ModuleList([CABlock(d_model, nhead, dropout) for _ in range(num_ca_layers)])

        # 预测支路专用（可学习 query + per-token 头，Stage 1 结束后丢弃）
        # p2m: 251 个代谢插槽与目标一一对应 -> 直接读头; m2p: 359 个蛋白插槽 -> 标量+映射
        self.meta_queries = nn.Parameter(torch.randn(1, meta_dim, d_model) * 0.02)
        self.prot_queries = nn.Parameter(torch.randn(1, num_prot_tokens, d_model) * 0.02)
        self.head_p2m = DirectPerTokenHead(d_model)                       # [B,251,d] -> x_meta [B,251]
        self.head_m2p = PerTokenPredictHead(d_model, num_prot_tokens, prot_dim, dropout=dropout)  # [B,359,d] -> x_prot
        self.head_recon_p = mlp_head(d_model, prot_dim, dropout=dropout)  # z_p -> x_prot
        self.head_recon_m = mlp_head(d_model, meta_dim, dropout=dropout)  # z_m -> x_meta

    def forward(self, x_prot, x_meta, pool=None):
        params = dict(self.named_parameters())
        use_pool = pool or self.pool
        B = x_prot.size(0)

        toks_p, gate = self.tokenizer(x_prot)           # [B, 359, d]
        toks_m = self.meta_encoder(x_meta)              # [B, 251, d]
        toks_p = F.dropout(toks_p, p=self.dropout_rate, training=self.training)
        toks_m = F.dropout(toks_m, p=self.dropout_rate, training=self.training)

        # ===== 表征支路（双边）=====
        q = toks_m
        for i in range(len(self.ca_layers)):
            q = self.ca_layers[i].functional_forward(q, toks_p, params, f'ca_layers.{i}')
        M_repr = q                                      # [B, 251, d]
        q = toks_p
        for i in range(len(self.ca_layers)):
            q = self.ca_layers[i].functional_forward(q, toks_m, params, f'ca_layers.{i}')
        P_repr = q                                      # [B, 359, d]
        z_m = pool_tokens(M_repr, use_pool)
        z_p = pool_tokens(P_repr, use_pool)

        # ===== 预测支路（单边，Q=可学习参数，KV=source 模态）-> per-token 头 =====
        q = self.meta_queries.expand(B, -1, -1)         # p2m: source=蛋白, target=代谢
        for i in range(len(self.ca_layers)):
            q = self.ca_layers[i].functional_forward(q, toks_p, params, f'ca_layers.{i}')
        pred_meta = self.head_p2m(q)                    # [B, 251]
        q = self.prot_queries.expand(B, -1, -1)         # m2p: source=代谢, target=蛋白
        for i in range(len(self.ca_layers)):
            q = self.ca_layers[i].functional_forward(q, toks_m, params, f'ca_layers.{i}')
        pred_prot = self.head_m2p(q)                    # [B, 2923]

        return {'pred_meta': pred_meta, 'pred_prot': pred_prot,
                'recon_prot': self.head_recon_p(z_p), 'recon_meta': self.head_recon_m(z_m),
                'gate': gate, 'z_prot': z_p, 'z_meta': z_m,
                'P_repr': P_repr, 'M_repr': M_repr}


class CLSDualCAMamlModel(nn.Module):
    """Stage 2: 原版风格 MAML。只用表征支路，inner loop 只更新 classifier + gate_logits。"""

    def __init__(self, num_proteins, num_metabolites, args, pathway_mask=None, unknown_indices=None):
        super().__init__()
        self.args = args
        d = args.embed_dim
        self.pool = getattr(args, 'pool', 'max')
        self.tokenizer = PathwayGatedTokenizer(num_proteins, d, pathway_mask, unknown_indices)
        self.meta_encoder = SimpleMetaEncoder(num_metabolites, d)
        self.ca_layers = nn.ModuleList([CABlock(d, args.num_heads, args.dropout_rate)
                                        for _ in range(args.num_layers)])
        self.shortcut_proj = nn.Linear(num_proteins, d)
        self.classifier = nn.Linear(2 * d, 1)
        self.dropout = nn.Dropout(args.dropout_rate)
        # 模态门控: z = concat(w_prot*z_p, w_meta*z_m), w = softmax(mod_gate)
        # 参数名含 'gate' -> inner loop 按疾病适配 (复用现有 grad_key_filter 机制)
        self.mod_gate = nn.Parameter(torch.zeros(2))

        # Meta-SGD: 每个参数一个可学习学习率
        self.alphas = nn.ParameterDict()
        for name, param in self.named_parameters():
            if 'alphas' not in name:
                self.alphas[name.replace('.', '_')] = nn.Parameter(torch.ones_like(param) * args.inner_lr)

    def functional_forward(self, x_prot, x_meta, params):
        p3d = x_prot.dim() == 3
        if p3d:
            Bt, N, P = x_prot.shape
            xp = x_prot.reshape(Bt * N, P)
            xm = x_meta.reshape(Bt * N, -1)
        else:
            xp, xm = x_prot, x_meta

        toks_p, gate = self.tokenizer.functional_forward(xp, params)
        toks_m = F.linear(xm.unsqueeze(-1), params['meta_encoder.token_embed.weight'],
                          params['meta_encoder.token_embed.bias']) + params['meta_encoder.pos_embed']
        dr = self.args.dropout_rate
        toks_p = F.dropout(toks_p, p=dr, training=self.training)
        toks_m = F.dropout(toks_m, p=dr, training=self.training)

        # 表征支路（双边 CA + pooling）
        q = toks_m
        for i in range(len(self.ca_layers)):
            q = self.ca_layers[i].functional_forward(q, toks_p, params, f'ca_layers.{i}')
        z_m = pool_tokens(q, self.pool)
        q = toks_p
        for i in range(len(self.ca_layers)):
            q = self.ca_layers[i].functional_forward(q, toks_m, params, f'ca_layers.{i}')
        z_p = pool_tokens(q, self.pool)

        z_p = F.dropout(z_p, p=dr, training=self.training)
        z_m = F.dropout(z_m, p=dr, training=self.training)
        if getattr(self.args, 'use_mod_gate', False):
            w = F.softmax(params['mod_gate'], dim=0)   # [2], params 兼容 fast_weights
            z_p = z_p * w[0]
            z_m = z_m * w[1]
        shortcut = F.linear(xp, params['shortcut_proj.weight'], params['shortcut_proj.bias'])
        shortcut = F.dropout(shortcut, p=dr, training=self.training)

        z = torch.cat([z_p + shortcut, z_m], dim=-1)          # [B, 128]
        logits = F.linear(z, params['classifier.weight'], params['classifier.bias'])
        if p3d:
            logits = logits.view(Bt, N, 1)
        return logits, gate

    def extract_cls(self, x_prot, x_meta):
        """提取表征支路 z_p/z_m (不过 mod_gate 缩放、不做分类), 供对比学习 loss 使用。"""
        params = dict(self.named_parameters())
        dr = self.args.dropout_rate
        toks_p, _ = self.tokenizer(x_prot)
        toks_m = self.meta_encoder(x_meta)
        toks_p = F.dropout(toks_p, p=dr, training=self.training)
        toks_m = F.dropout(toks_m, p=dr, training=self.training)

        q = toks_m
        for i in range(len(self.ca_layers)):
            q = self.ca_layers[i].functional_forward(q, toks_p, params, f'ca_layers.{i}')
        z_m = pool_tokens(q, self.pool)
        q = toks_p
        for i in range(len(self.ca_layers)):
            q = self.ca_layers[i].functional_forward(q, toks_m, params, f'ca_layers.{i}')
        z_p = pool_tokens(q, self.pool)

        z_p = F.dropout(z_p, p=dr, training=self.training)
        z_m = F.dropout(z_m, p=dr, training=self.training)
        return z_p, z_m


def load_pretrained_encoders(model, ckpt_path):
    """Stage 1 ckpt -> Stage 2 模型: 只加载共享 encoder, 丢弃 query 和 4 个头"""
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ck['model_state_dict'] if 'model_state_dict' in ck else ck
    mapping = {k: v for k, v in sd.items()
               if not k.startswith('head_') and k not in ('meta_queries', 'prot_queries')}
    missing, unexpected = model.load_state_dict(mapping, strict=False)
    n_loaded = len(mapping) - len(unexpected)
    print(f"[warm-start cls-ca] loaded {n_loaded} tensors; missing={len(missing)}, unexpected={len(unexpected)}")
