# -*- coding: utf-8 -*-
"""机制一 CLSDual: 双分支 transformer + CLS 拼接。

整体结构 (Stage 1 预训练 / Stage 2 MAML 共用同一套 encoder, 权重名一致可热启动):

  x_prot [B,2923]
    -> PathwayGatedTokenizer          # 327 通路 token + 32 unk token = 359 token
    -> 前面拼一个可学习 cls_token     # 360 token
    -> 2 x MetaTransformerLayer (自注意力)
    -> 取第 0 位 = cls_prot [B,64]    # 蛋白分支的全局表征

  x_meta [B,251]
    -> SimpleMetaEncoder              # 251 个 token
    -> 前面拼一个可学习 meta_cls_token
    -> 2 x MetaTransformerLayer (独立参数)
    -> 取第 0 位 = cls_meta [B,64]    # 代谢分支的全局表征

Stage 1 (PretrainCLSDual): 4 个辅助头做相互预测 + 自重建 (训练完丢弃):
  head_p2m: 蛋白 token 矩阵 -> 预测代谢谱 x_meta   (per-token 头)
  head_m2p: 代谢 token 矩阵 -> 预测蛋白谱 x_prot   (per-token 头)
  head_recon_p: cls_prot -> 重建 x_prot           (MLP)
  head_recon_m: cls_meta -> 重建 x_meta           (MLP)

Stage 2 (CLSDualMamlModel): 拼接做疾病分类:
  z = concat( cls_prot + shortcut_proj(x_prot),  cls_meta )   # [B,128]
  logits = Linear(128 -> 1)(z)
  shortcut 是原版 ProMeta 的"抄近路": 原始蛋白谱直接线性压缩到 64 维,
  防止信息在 tokenizer/attention 里走丢。
  [可选] mod_gate: softmax 标量对 (w_prot, w_meta) 先缩放两个 CLS 再拼接,
        让 inner loop 能按疾病调融合权重 (实验结论: 会塌缩, 见 docs/RESULTS.md)。

inner loop 只更新参数名含 classifier/gate 的 (分类头 + 蛋白 gate_logits [+ mod_gate]),
encoder 主体由 outer loop 更新。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from prometa201.omics_encoders import PathwayGatedTokenizer, SimpleMetaEncoder
from prometa201.layers import MetaTransformerLayer


def mlp_head(in_dim, out_dim, hidden=128, dropout=0.5):
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
        nn.Dropout(dropout), nn.Linear(hidden, out_dim))


class PerTokenPredictHead(nn.Module):
    """per-token 预测头: token 矩阵 [B,N,d] 逐位置读标量 [B,N] -> 全连接映射到目标谱。

    无池化瓶颈: 第 i 个 slot 直接参与预测第 i 个目标值 (对齐 Plus_v2/bbbPrometa 的标准做法)。
    p2m: N=蛋白token数 -> out=251;  m2p: N=代谢token数 -> out=2923。
    """

    def __init__(self, d_model, n_tokens, out_dim, dropout=0.5):
        super().__init__()
        self.scalar = nn.Linear(d_model, 1)
        self.map = nn.Sequential(
            nn.Linear(n_tokens, n_tokens * 4), nn.LayerNorm(n_tokens * 4), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(n_tokens * 4, out_dim))

    def forward(self, tokens):
        s = self.scalar(tokens).squeeze(-1)      # [B, N]
        return self.map(s)                        # [B, out_dim]


class DirectPerTokenHead(nn.Module):
    """per-token 直接读头: N 个 slot 与目标维度一一对应 (N==out_dim), 无映射层。

    用于 CA 预测支路: 251 个代谢插槽各吐一个值 -> [B,251]。
    """

    def __init__(self, d_model):
        super().__init__()
        self.scalar = nn.Linear(d_model, 1)

    def forward(self, tokens):
        return self.scalar(tokens).squeeze(-1)   # [B, N]


class PretrainCLSDual(nn.Module):
    """Stage 1: CLS 级相互预测 + 自重建预训练"""

    def __init__(self, meta_dim=251, prot_dim=2923, d_model=64, nhead=2,
                 num_layers=2, dropout=0.5, pathway_mask=None, unknown_indices=None):
        super().__init__()
        # 双分支 encoder (与 Stage 2 模型同名, 权重直接互通)
        self.tokenizer = PathwayGatedTokenizer(prot_dim, d_model, pathway_mask, unknown_indices)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.prot_tf = nn.ModuleList([MetaTransformerLayer(d_model, nhead, dropout) for _ in range(num_layers)])

        self.meta_encoder = SimpleMetaEncoder(meta_dim, d_model)
        self.meta_cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.meta_tf = nn.ModuleList([MetaTransformerLayer(d_model, nhead, dropout) for _ in range(num_layers)])

        # 4 个预测/重建头 (Stage 1 专用, 下游丢弃)
        # unk token 数 = PathwayGatedTokenizer.hidden_mult(=32), 与 unknown_indices 长度无关
        num_prot_tokens = pathway_mask.size(0) + (32 if unknown_indices else 0)
        self.head_p2m = PerTokenPredictHead(d_model, num_prot_tokens, meta_dim, dropout=dropout)  # 蛋白token -> x_meta
        self.head_m2p = PerTokenPredictHead(d_model, meta_dim, prot_dim, dropout=dropout)         # 代谢token -> x_prot
        self.head_recon_p = mlp_head(d_model, prot_dim, dropout=dropout)  # cls_prot -> x_prot
        self.head_recon_m = mlp_head(d_model, meta_dim, dropout=dropout)  # cls_meta -> x_meta

    def forward(self, x_prot, x_meta):
        params = dict(self.named_parameters())   # 标准训练: 直接用模块自身参数
        toks_p, gate = self.tokenizer(x_prot)
        seq_p = torch.cat([self.cls_token.expand(toks_p.size(0), -1, -1), toks_p], dim=1)
        seq_p = F.dropout(seq_p, p=0.5, training=self.training)
        for i in range(len(self.prot_tf)):
            seq_p = self.prot_tf[i].functional_forward(seq_p, params, f'prot_tf.{i}')
        cls_p = seq_p[:, 0]

        toks_m = self.meta_encoder(x_meta)
        seq_m = torch.cat([self.meta_cls_token.expand(toks_m.size(0), -1, -1), toks_m], dim=1)
        seq_m = F.dropout(seq_m, p=0.5, training=self.training)
        for i in range(len(self.meta_tf)):
            seq_m = self.meta_tf[i].functional_forward(seq_m, params, f'meta_tf.{i}')
        cls_m = seq_m[:, 0]

        return {'pred_meta': self.head_p2m(seq_p[:, 1:]), 'pred_prot': self.head_m2p(seq_m[:, 1:]),
                'recon_prot': self.head_recon_p(cls_p), 'recon_meta': self.head_recon_m(cls_m),
                'gate': gate, 'cls_prot': cls_p, 'cls_meta': cls_m}


class CLSDualMamlModel(nn.Module):
    """Stage 2: 原版风格 MAML。inner loop 只更新 classifier + gate_logits [+ mod_gate]。"""

    def __init__(self, num_proteins, num_metabolites, args, pathway_mask=None, unknown_indices=None):
        super().__init__()
        self.args = args
        d = args.embed_dim
        self.tokenizer = PathwayGatedTokenizer(num_proteins, d, pathway_mask, unknown_indices)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d))
        self.prot_tf = nn.ModuleList([MetaTransformerLayer(d, args.num_heads, args.dropout_rate)
                                      for _ in range(args.num_layers)])
        self.meta_encoder = SimpleMetaEncoder(num_metabolites, d)
        self.meta_cls_token = nn.Parameter(torch.randn(1, 1, d))
        self.meta_tf = nn.ModuleList([MetaTransformerLayer(d, args.num_heads, args.dropout_rate)
                                      for _ in range(args.num_layers)])
        self.shortcut_proj = nn.Linear(num_proteins, d)
        self.classifier = nn.Linear(2 * d, 1)
        self.dropout = nn.Dropout(args.dropout_rate)
        # 模态门控: z = concat(w_prot*cls_p, w_meta*cls_m), w = softmax(mod_gate)
        # 参数名含 'gate' -> inner loop 按疾病适配 (复用现有 grad_key_filter 机制)
        self.mod_gate = nn.Parameter(torch.zeros(2))

        # Meta-SGD: 每个参数一个可学习学习率 (inner loop 的梯度步长)
        self.alphas = nn.ParameterDict()
        for name, param in self.named_parameters():
            if 'alphas' not in name:
                self.alphas[name.replace('.', '_')] = nn.Parameter(torch.ones_like(param) * args.inner_lr)

    def functional_forward(self, x_prot, x_meta, params):
        """支持 MAML 的前向: params 传 fast_weights (inner-adapt 后的临时参数)。"""
        p3d = x_prot.dim() == 3
        if p3d:
            Bt, N, P = x_prot.shape
            xp = x_prot.reshape(Bt * N, P)
            xm = x_meta.reshape(Bt * N, -1)
        else:
            xp, xm = x_prot, x_meta

        # ---- 蛋白分支: tokenizer -> +cls_token -> 2层 transformer -> cls_prot ----
        toks_p, gate = self.tokenizer.functional_forward(xp, params)
        cls_tok = params['cls_token']
        seq_p = torch.cat([cls_tok.expand(toks_p.size(0), -1, -1), toks_p], dim=1)
        seq_p = F.dropout(seq_p, p=self.args.dropout_rate, training=self.training)
        for i in range(len(self.prot_tf)):
            seq_p = self.prot_tf[i].functional_forward(seq_p, params, prefix=f'prot_tf.{i}')
        cls_p = seq_p[:, 0]

        # ---- 代谢分支: encoder -> +meta_cls_token -> 2层 transformer -> cls_meta ----
        toks_m = F.linear(xm.unsqueeze(-1), params['meta_encoder.token_embed.weight'],
                          params['meta_encoder.token_embed.bias']) + params['meta_encoder.pos_embed']
        mcls = params['meta_cls_token']
        seq_m = torch.cat([mcls.expand(toks_m.size(0), -1, -1), toks_m], dim=1)
        seq_m = F.dropout(seq_m, p=self.args.dropout_rate, training=self.training)
        for i in range(len(self.meta_tf)):
            seq_m = self.meta_tf[i].functional_forward(seq_m, params, prefix=f'meta_tf.{i}')
        cls_m = seq_m[:, 0]

        cls_p = F.dropout(cls_p, p=self.args.dropout_rate, training=self.training)
        cls_m = F.dropout(cls_m, p=self.args.dropout_rate, training=self.training)
        if getattr(self.args, 'use_mod_gate', False):
            w = F.softmax(params['mod_gate'], dim=0)   # [2], params 兼容 fast_weights
            cls_p = cls_p * w[0]
            cls_m = cls_m * w[1]
        shortcut = F.linear(xp, params['shortcut_proj.weight'], params['shortcut_proj.bias'])
        shortcut = F.dropout(shortcut, p=self.args.dropout_rate, training=self.training)

        z = torch.cat([cls_p + shortcut, cls_m], dim=-1)          # [B, 128]
        logits = F.linear(z, params['classifier.weight'], params['classifier.bias'])
        if p3d:
            logits = logits.view(Bt, N, 1)
        return logits, gate

    def extract_cls(self, x_prot, x_meta):
        """提取双分支 CLS (不做分类), 供对比学习 loss 使用。编码路径与 functional_forward 一致。"""
        params = dict(self.named_parameters())
        toks_p, _ = self.tokenizer(x_prot)
        seq_p = torch.cat([self.cls_token.expand(toks_p.size(0), -1, -1), toks_p], dim=1)
        seq_p = F.dropout(seq_p, p=self.args.dropout_rate, training=self.training)
        for i in range(len(self.prot_tf)):
            seq_p = self.prot_tf[i].functional_forward(seq_p, params, f'prot_tf.{i}')
        cls_p = seq_p[:, 0]

        toks_m = self.meta_encoder(x_meta)
        seq_m = torch.cat([self.meta_cls_token.expand(toks_m.size(0), -1, -1), toks_m], dim=1)
        seq_m = F.dropout(seq_m, p=self.args.dropout_rate, training=self.training)
        for i in range(len(self.meta_tf)):
            seq_m = self.meta_tf[i].functional_forward(seq_m, params, f'meta_tf.{i}')
        cls_m = seq_m[:, 0]

        cls_p = F.dropout(cls_p, p=self.args.dropout_rate, training=self.training)
        cls_m = F.dropout(cls_m, p=self.args.dropout_rate, training=self.training)
        return cls_p, cls_m


def load_pretrained_encoders(model, ckpt_path):
    """Stage 1 ckpt -> Stage 2 模型: 只加载双分支 encoder, 丢弃 4 个头"""
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ck['model_state_dict'] if 'model_state_dict' in ck else ck
    mapping = {k: v for k, v in sd.items() if not k.startswith('head_')}
    missing, unexpected = model.load_state_dict(mapping, strict=False)
    n_loaded = len(mapping) - len(unexpected)
    print(f"[warm-start cls-dual] loaded {n_loaded} tensors; missing={len(missing)}, unexpected={len(unexpected)}")
