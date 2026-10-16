# -*- coding: utf-8 -*-
"""RQ2 代谢物重要性 (单变量 AUROC 版): 每个疾病 x 每个代谢物, 在 query 集上算单变量 AUROC。

背景: 置换重要性在融合模型上失效 —— 两类融合架构的 per-disease 分类器实际忽略代谢支
(全部置零 AUROC 变化 <0.01), 所以"哪些代谢物有信息"改用与融合模型无关的单变量 AUROC,
直接回答"每个疾病的代谢组信号集中在哪些代谢物上"。

输出 meta_univariate.npz: auc [91, 251], terms, meta_names (query 集与 RQ2 提取完全同 split)

用法: python prometa201/analysis/meta_univariate_auc.py
"""
import os
import sys
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from prometa201.paths import PREPROCESSED_V2_DIR, PKL_INPUT_DIR, OUTPUT_DIR
from prometa201.dataset import DualOmicsDataset
from sklearn.metrics import roc_auc_score

OUT_DIR = os.path.join(OUTPUT_DIR, 'analysis')
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    eids = np.concatenate([np.load(os.path.join(PREPROCESSED_V2_DIR, 'eids_train.npy')),
                           np.load(os.path.join(PREPROCESSED_V2_DIR, 'eids_val.npy'))], axis=0)
    X_meta = np.nan_to_num(np.concatenate([np.load(os.path.join(PREPROCESSED_V2_DIR, 'X_meta_train.npy')),
                                           np.load(os.path.join(PREPROCESSED_V2_DIR, 'X_meta_val.npy'))], axis=0).astype(np.float32))
    X_prot = np.nan_to_num(np.concatenate([np.load(os.path.join(PREPROCESSED_V2_DIR, 'X_prot_train.npy')),
                                           np.load(os.path.join(PREPROCESSED_V2_DIR, 'X_prot_val.npy'))], axis=0).astype(np.float32))
    eid_to_idx = {str(int(e)): i for i, e in enumerate(eids)}
    with open(os.path.join(PKL_INPUT_DIR, 'term2pre_cases_test.pkl'), 'rb') as f:
        test_case = pickle.load(f)
    with open(os.path.join(PKL_INPUT_DIR, 'term2pre_controls_test.pkl'), 'rb') as f:
        test_ctrl = pickle.load(f)

    ds = DualOmicsDataset(X_prot, X_meta, test_case, test_ctrl, eid_to_idx,
                          support_size=32, mode='test', random_seed=42)
    terms, aucs = [], []
    n_meta = X_meta.shape[1]
    for i in range(len(ds)):
        qX_prot, qX_meta, qY, _, _, _, term = ds[i]
        y = qY.numpy()
        if len(np.unique(y)) < 2:
            continue
        x = qX_meta.numpy()
        a = np.full(n_meta, np.nan)
        for j in range(n_meta):
            if np.std(x[:, j]) > 0:
                a[j] = roc_auc_score(y, x[:, j])
        terms.append(str(term)); aucs.append(a)
        if (i + 1) % 20 == 0:
            print(f'[{i+1}/{len(ds)}]', flush=True)
    meta_names = open(os.path.join(PREPROCESSED_V2_DIR, 'meta_names.txt')).read().strip().split('\n')
    np.savez(os.path.join(OUT_DIR, 'meta_univariate.npz'),
             terms=np.array(terms), auc=np.stack(aucs),
             meta_names=np.array(meta_names))
    print('saved:', os.path.join(OUT_DIR, 'meta_univariate.npz'), flush=True)


if __name__ == '__main__':
    main()
