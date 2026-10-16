# -*- coding: utf-8 -*-
"""输入端 tokenizer — 把原始组学向量变成 transformer 的 token 序列。

三个组件均来自原版 ProMeta 代码库 (Prometa2.0_clean_split/transformer/models.py):

  PathwayGatedTokenizer  蛋白侧: 2923 个蛋白 -> 327 个 KEGG 通路 token + 32 个 unk token。
      核心两步: (1) 每个蛋白学一个 sigmoid 门控 gate∈[0,1]; (2) 用通路掩码矩阵做
      加权求和(einsum), 把同通路的蛋白聚成一个 token。不在任何通路的蛋白
      单独走一个小 MLP 变成 32 个 "unknown" token。
  SimpleMetaEncoder      代谢侧: 251 个代谢物 -> 251 个 token (每个代谢物一个
      Linear(1->d) + 位置编码)。代谢物没有通路注释可用, 一一对应最简单。
  CrossAttentionBlock    交叉注意力 block (nn.MultiheadAttention 封装), 供
      DualOmicsSharedModel(BCA 原型)使用; CLSCA 用的是 models_cls_ca.CABlock。

另含 load_pathway_mask: 从 CPDB 注释文件构建 通路×蛋白 的 0/1 掩码矩阵。
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class PathwayGatedTokenizer(nn.Module):
    """原版 ProMeta 的 Pathway-Gated Tokenizer (直接复用核心思路)

    输入 [B, num_proteins] -> 输出 token 序列 [B, n_pathways+32, embed_dim] 和 gate [num_proteins]
    """

    def __init__(self, num_proteins, embed_dim, pathway_mask, unknown_indices):
        super().__init__()
        self.num_proteins = num_proteins
        self.embed_dim = embed_dim
        self.unknown_indices = unknown_indices

        self.mid_dim = 32
        self.shared_linear = nn.Linear(1, self.mid_dim)              # 每个蛋白的表达值 -> 32 维
        self.pos_embedding = nn.Parameter(torch.randn(1, num_proteins, self.mid_dim))
        self.gate_logits = nn.Parameter(torch.zeros(num_proteins))   # 每蛋白一个可学习门控
        self.register_buffer('pathway_mask', pathway_mask)           # [n_pathways, num_proteins] 0/1

        self.out_projector = nn.Linear(self.mid_dim, embed_dim)

        self.num_unknown = len(unknown_indices)
        self.hidden_mult = 32

        if self.num_unknown > 0:
            self.unknown_mlp = nn.Linear(self.num_unknown, self.hidden_mult * embed_dim)
            self.unknown_ln = nn.LayerNorm(embed_dim)

    def functional_forward(self, x, params):
        """支持 MAML 的参数传递版前向 (params 替代 self.parameters())"""
        B = x.shape[0]
        x_reshaped = x.unsqueeze(-1)

        w_shared = params['tokenizer.shared_linear.weight']
        b_shared = params['tokenizer.shared_linear.bias']
        x_emb = F.linear(x_reshaped, w_shared, b_shared)

        p_emb = params['tokenizer.pos_embedding']
        x_emb = x_emb + p_emb

        g_logits = params['tokenizer.gate_logits']
        gate = torch.sigmoid(g_logits)                                # [num_proteins], 每蛋白门控
        x_gated = x_emb * gate.view(1, -1, 1)

        # 通路聚合: einsum('kn,bnd->bkd') = 掩码加权和, 同一通路的蛋白聚成一个 token
        pathway_tokens = torch.einsum('kn, bnd -> bkd', self.pathway_mask, x_gated)

        w_out = params['tokenizer.out_projector.weight']
        b_out = params['tokenizer.out_projector.bias']
        pathway_tokens = F.linear(pathway_tokens, w_out, b_out)

        final_tokens = pathway_tokens
        if self.num_unknown > 0:
            # 不属于任何通路的蛋白 -> 摊平后过 MLP -> 32 个 unk token
            x_unk = x[:, self.unknown_indices]
            w_unk = params['tokenizer.unknown_mlp.weight']
            b_unk = params['tokenizer.unknown_mlp.bias']

            unk_features = F.linear(x_unk, w_unk, b_unk)
            unk_tokens = unk_features.view(B, self.hidden_mult, self.embed_dim)

            ln_w = params.get('tokenizer.unknown_ln.weight', self.unknown_ln.weight)
            ln_b = params.get('tokenizer.unknown_ln.bias', self.unknown_ln.bias)
            unk_tokens = F.layer_norm(unk_tokens, (self.embed_dim,), ln_w, ln_b)

            final_tokens = torch.cat([pathway_tokens, unk_tokens], dim=1)

        return final_tokens, gate

    def forward(self, x):
        """标准前向 (非 MAML)"""
        B = x.shape[0]
        x_reshaped = x.unsqueeze(-1)

        x_emb = self.shared_linear(x_reshaped)  # [B, 2923, 32]
        x_emb = x_emb + self.pos_embedding      # [B, 2923, 32]

        gate = torch.sigmoid(self.gate_logits)  # [2923]
        x_gated = x_emb * gate.view(1, -1, 1)   # [B, 2923, 32]

        pathway_tokens = torch.einsum('kn, bnd -> bkd',
                                      self.pathway_mask, x_gated)  # [B, P, 32]
        pathway_tokens = self.out_projector(pathway_tokens)         # [B, P, embed_dim]

        if self.num_unknown > 0:
            x_unk = x[:, self.unknown_indices]      # [B, num_unknown]
            unk_features = self.unknown_mlp(x_unk)  # [B, 32*embed_dim]
            unk_tokens = unk_features.view(B, self.hidden_mult, self.embed_dim)
            unk_tokens = self.unknown_ln(unk_tokens)
            final_tokens = torch.cat([pathway_tokens, unk_tokens], dim=1)
        else:
            final_tokens = pathway_tokens

        return final_tokens, gate


class SimpleMetaEncoder(nn.Module):
    """代谢物侧 tokenizer: [B, 251] -> [B, 251, d_model]

    每个代谢物一个标量值, Linear(1->d) 升维 + 位置编码, 一一对应成 251 个 token。
    """

    def __init__(self, meta_dim=251, d_model=128):
        super().__init__()
        self.token_embed = nn.Linear(1, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, meta_dim, d_model))

    def forward(self, x):
        tokens = self.token_embed(x.unsqueeze(-1))  # [B, 251, d_model]
        tokens = tokens + self.pos_embed             # [B, 251, d_model]
        return tokens


class CrossAttentionBlock(nn.Module):
    """Cross Attention Block + FFN + LayerNorm (nn.MultiheadAttention 封装)

    forward(query, kv): Q 来自 query, K/V 来自 kv。
    供 DualOmicsSharedModel(BCA 共享空间原型)使用; CLSCA 见 models_cls_ca.CABlock。
    """

    def __init__(self, d_model=128, nhead=4, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead,
            dropout=dropout, batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, kv, need_weights=False):
        attn_output, attn_weights = self.cross_attn(
            query=query, key=kv, value=kv,
            need_weights=need_weights,
            average_attn_weights=True,
        )
        query = self.norm1(query + self.dropout(attn_output))
        ff_output = self.ffn(query)
        query = self.norm2(query + self.dropout(ff_output))
        return query, attn_weights


def load_pathway_mask(cpdb_path, protein_names):
    """从 CPDB 注释文件构建 通路×蛋白 0/1 掩码矩阵 (只用 KEGG 来源)。

    返回 (mask [n_kegg_pathways, n_proteins] tensor, unknown_indices [不在任何通路的蛋白下标])。
    剔除没有任何蛋白命中的空通路后, 本数据上为 327 条通路。
    """
    protein2id = {str(p).strip().upper().split('.')[0]: i for i, p in enumerate(protein_names)}
    cpdb = pd.read_csv(cpdb_path, sep='\t', dtype=str)
    gene_col = 'hgnc_symbol_ids' if 'hgnc_symbol_ids' in cpdb.columns else 'hgnc_symbol'
    cpdb = cpdb.dropna(subset=[gene_col]).drop_duplicates(subset=['pathway', 'source'])
    cpdb = cpdb[cpdb['source'] == 'KEGG']
    pathways = cpdb['pathway'].unique()
    local2id = {l: i for i, l in enumerate(pathways)}
    mask = np.zeros((len(pathways), len(protein_names)), dtype=np.float32)
    for local, genes_str in zip(cpdb['pathway'].values, cpdb[gene_col].values):
        for gene in genes_str.split(','):
            gc = gene.strip().upper()
            if gc in protein2id:
                mask[local2id[local], protein2id[gc]] = 1.0
    mask = mask[mask.sum(axis=1) > 0]
    unknown_indices = np.where(mask.sum(axis=0) == 0)[0].tolist()
    return torch.from_numpy(mask), unknown_indices
