# -*- coding: utf-8 -*-
"""基础网络层 — 从原版 ProMeta 代码库原样 vendored 的两个组件。

来源:
  - FocalLoss / MetaTransformerLayer: prometa_reproduce_dualomics/model.py (原版双组学复现)

MetaTransformerLayer 只有 functional_forward(没有标准 forward),
因为 MAML inner loop 需要把"临时更新后的参数"显式传进每一层。
结构 = 手写 Pre-LN 风格 transformer block: 多头自注意力 + GELU FFN, 残差连接。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """二分类 Focal Loss:  down-weight 容易样本, 聚焦难样本。

    L = alpha * (1 - p_t)^gamma * BCE(x, y)
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss


class MetaTransformerLayer(nn.Module):
    """Functional Transformer Layer supporting meta-learning parameter passing.

    唯一对外接口是 functional_forward(x, params, prefix):
      params 是 {参数名: 张量} 字典 —— 标准训练时传自身 named_parameters(),
      MAML inner loop 时传 inner-update 之后的 fast_weights。
    """

    def __init__(self, embed_dim, num_heads, dropout=0.4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.linear1 = nn.Linear(embed_dim, embed_dim * 4)
        self.linear2 = nn.Linear(embed_dim * 4, embed_dim)

        self.dropout = dropout

    def functional_forward(self, x, params, prefix):
        B, L, D = x.shape
        residual = x

        # --- 多头自注意力 (q/k/v 投影 -> 分头 -> 缩放点积 -> 拼头 -> 输出投影) ---
        q = F.linear(x, params[f'{prefix}.q_proj.weight'], params[f'{prefix}.q_proj.bias'])
        k = F.linear(x, params[f'{prefix}.k_proj.weight'], params[f'{prefix}.k_proj.bias'])
        v = F.linear(x, params[f'{prefix}.v_proj.weight'], params[f'{prefix}.v_proj.bias'])

        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, D)
        out = F.linear(out, params[f'{prefix}.out_proj.weight'], params[f'{prefix}.out_proj.bias'])

        x = residual + F.dropout(out, p=self.dropout, training=self.training)
        residual = x

        # --- FFN: Linear -> GELU -> Linear (4x 隐层) ---
        ff = F.linear(x, params[f'{prefix}.linear1.weight'], params[f'{prefix}.linear1.bias'])
        ff = F.gelu(ff)
        ff = F.dropout(ff, p=self.dropout, training=self.training)
        ff = F.linear(ff, params[f'{prefix}.linear2.weight'], params[f'{prefix}.linear2.bias'])
        ff = F.dropout(ff, p=self.dropout, training=self.training)

        x = residual + ff
        return x
