# -*- coding: utf-8 -*-
"""Few-shot 任务数据集: 每个"疾病"是一个二分类任务 (病例 vs 对照)。

一个样本 = (query批, support批, 疾病名):
  - support: 该疾病 support_size 个带标签样本 (32 = 16病例+16对照),
             MAML inner loop 在上面临时更新分类头
  - query:   最多 query_size 个样本 (默认 128), inner-adapt 后的模型在上面算 loss / 指标

split 方式: 每个疾病的病例/对照列表按 seed 固定的随机种子 shuffle 后切两段,
前 max_support_size 个留作 support 池, 其余作 query 池 —— support/query 不重叠。
"""
import numpy as np
import torch
from torch.utils.data import Dataset


class DualOmicsDataset(Dataset):
    def __init__(self, prot_features, meta_features, case_dict, control_dict,
                 eid_to_idx, support_size=32, max_support_size=32, query_size=128,
                 mode='train', random_seed=42):
        self.prot_features = prot_features
        self.meta_features = meta_features
        self.eid_to_idx = eid_to_idx
        self.support_size = support_size
        self.query_size = query_size
        self.mode = mode
        self.seed = random_seed

        valid_terms = [t for t in case_dict.keys() if t in control_dict]
        self.term2support = {}
        self.term2query = {}
        self.tasks = []
        for i, term in enumerate(valid_terms):
            case_indices = [eid_to_idx[str(e)] for e in case_dict[term] if str(e) in eid_to_idx]
            ctrl_indices = [eid_to_idx[str(e)] for e in control_dict[term] if str(e) in eid_to_idx]
            rng = np.random.RandomState(self.seed + i)
            rng.shuffle(case_indices); rng.shuffle(ctrl_indices)
            try:
                sup_c = case_indices[:max_support_size//2][:self.support_size//2]
                sup_t = ctrl_indices[:max_support_size//2][:self.support_size//2]
                if len(sup_c) == 0 or len(sup_t) == 0:
                    continue
                case_indices = case_indices[max_support_size//2:]
                ctrl_indices = ctrl_indices[max_support_size//2:]
                n_case = min(len(case_indices), query_size//4)
                n_ctrl = min(len(ctrl_indices), query_size - n_case)
                q_c = case_indices[:n_case]; q_t = ctrl_indices[:n_ctrl]
                if len(q_c) == 0 or len(q_t) == 0:
                    continue
                self.term2support[term] = (sup_c, sup_t)
                self.term2query[term] = (q_c, q_t)
                self.tasks.append(term)
            except Exception:
                continue
        print(f"[{mode.upper()} Dataset] Prepared support/query splits for {len(self.tasks)} terms.")

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        term = self.tasks[idx]
        sup_c, sup_t = self.term2support[term]
        q_c, q_t = self.term2query[term]
        si = np.concatenate([sup_c, sup_t]); qi = np.concatenate([q_c, q_t])
        sX_prot = torch.tensor(self.prot_features[si], dtype=torch.float32)
        qX_prot = torch.tensor(self.prot_features[qi], dtype=torch.float32)
        sX_meta = torch.tensor(self.meta_features[si], dtype=torch.float32)
        qX_meta = torch.tensor(self.meta_features[qi], dtype=torch.float32)
        sY = torch.tensor(np.concatenate([np.ones(len(sup_c)), np.zeros(len(sup_t))]), dtype=torch.float32)
        qY = torch.tensor(np.concatenate([np.ones(len(q_c)), np.zeros(len(q_t))]), dtype=torch.float32)
        return qX_prot, qX_meta, qY, sX_prot, sX_meta, sY, term
