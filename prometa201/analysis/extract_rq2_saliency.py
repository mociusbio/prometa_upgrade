# -*- coding: utf-8 -*-
"""RQ2 蛋白/代谢物 逐疾病 saliency (输入梯度) 提取。

背景: tokenizer 的蛋白 gate 是全局参数, 所有疾病共享, 无法逐病比较 (extract_rq2.py)。
本脚本改逐疾病算输入梯度:
  每个疾病 inner-adapt 后, 对 query 样本的正类 logit 求关于
  原始输入特征 (x_prot / x_meta) 的梯度, 取 |grad| 在样本维的均值
  -> saliency_prot [91, 2923], saliency_meta [91, 251]

分析建议: 原始 saliency 有 ~70% 跨病共享成分, 逐病比较前先减去跨病均值。

用法:
  python prometa201/analysis/extract_rq2_saliency.py --ckpt <best.pth> --gpu_id 0 \
      --out rq2_saliency_seed42
"""
import os
import sys
import argparse
import pickle
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from prometa201.paths import (PREPROCESSED_V2_DIR, PKL_INPUT_DIR, CPDB_FILE_PATH,
                              PROT_NAMES_FILE, OUTPUT_DIR)
from prometa201.omics_encoders import load_pathway_mask
from prometa201.models_cls_dual import CLSDualMamlModel
from prometa201.dataset import DualOmicsDataset
from prometa201.maml import grad_key_filter, compute_task_metrics
from prometa201.analysis.extract_rq2 import Args, adapt

OUT_DIR = os.path.join(OUTPUT_DIR, 'analysis')
os.makedirs(OUT_DIR, exist_ok=True)


def saliency(model, q_prot, q_meta, fast_weights):
    """query 正类 logit 对输入特征的 |grad| 均值。返回 (prot_sal [P], meta_sal [M])"""
    p = q_prot.clone().detach().requires_grad_(True)
    m = q_meta.clone().detach().requires_grad_(True)
    logits, _ = model.functional_forward(p, m, fast_weights)
    logits.sum().backward()
    return p.grad.abs().mean(dim=1).squeeze(0).cpu().numpy(), \
           m.grad.abs().mean(dim=1).squeeze(0).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--gpu_id', type=int, default=0)
    ap.add_argument('--random_seed', type=int, default=42)
    ap.add_argument('--max_tasks', type=int, default=0)
    ap.add_argument('--out', default='rq2_saliency_seed42')
    cli = ap.parse_args()

    args = Args()
    args.random_seed = cli.random_seed
    device = torch.device(f'cuda:{cli.gpu_id}')

    eids = np.concatenate([np.load(os.path.join(PREPROCESSED_V2_DIR, 'eids_train.npy')),
                           np.load(os.path.join(PREPROCESSED_V2_DIR, 'eids_val.npy'))], axis=0)
    X_prot = np.concatenate([np.load(os.path.join(PREPROCESSED_V2_DIR, 'X_prot_train.npy')),
                             np.load(os.path.join(PREPROCESSED_V2_DIR, 'X_prot_val.npy'))], axis=0).astype(np.float32)
    X_meta = np.concatenate([np.load(os.path.join(PREPROCESSED_V2_DIR, 'X_meta_train.npy')),
                             np.load(os.path.join(PREPROCESSED_V2_DIR, 'X_meta_val.npy'))], axis=0).astype(np.float32)
    X_prot = np.nan_to_num(X_prot); X_meta = np.nan_to_num(X_meta)
    eid_to_idx = {str(int(e)): i for i, e in enumerate(eids)}

    with open(os.path.join(PKL_INPUT_DIR, 'term2pre_cases_test.pkl'), 'rb') as f:
        test_case = pickle.load(f)
    with open(os.path.join(PKL_INPUT_DIR, 'term2pre_controls_test.pkl'), 'rb') as f:
        test_ctrl = pickle.load(f)

    prot_names = open(PROT_NAMES_FILE).read().strip().split('\n')
    pathway_mask, unknown_indices = load_pathway_mask(CPDB_FILE_PATH, prot_names)
    pathway_mask = pathway_mask.to(device)

    test_dataset = DualOmicsDataset(X_prot, X_meta, test_case, test_ctrl, eid_to_idx,
                                    support_size=args.support_size, mode='test',
                                    random_seed=args.random_seed)
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = CLSDualMamlModel(num_proteins=X_prot.shape[1], num_metabolites=X_meta.shape[1],
                             args=args, pathway_mask=pathway_mask,
                             unknown_indices=unknown_indices).to(device)
    sd = torch.load(cli.ckpt, map_location='cpu', weights_only=False)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f'ckpt loaded: missing={len(missing)} unexpected={len(unexpected)}', flush=True)
    model.eval()

    n_total = min(cli.max_tasks, len(test_loader)) if cli.max_tasks > 0 else len(test_loader)
    terms, sal_p, sal_m, auroc_all = [], [], [], []
    for i, batch in enumerate(test_loader):
        if i >= n_total:
            break
        q_prot, q_meta, q_lb = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        s_prot, s_meta, s_lb = batch[3].to(device), batch[4].to(device), batch[5].to(device)
        term = batch[6][0]
        if len(torch.unique(s_lb[0])) < 2:
            continue
        fast_weights = adapt(model, s_prot, s_meta, s_lb, args, device)
        with torch.no_grad():
            logits, _ = model.functional_forward(q_prot, q_meta, fast_weights)
            probs = torch.sigmoid(logits).squeeze(0)
            if torch.isnan(probs).any() or len(torch.unique(q_lb[0])) < 2:
                continue
            m = compute_task_metrics(probs, q_lb.squeeze(0))
            if m is None:
                continue
        sp, sm = saliency(model, q_prot, q_meta, fast_weights)
        terms.append(str(term)); sal_p.append(sp); sal_m.append(sm); auroc_all.append(m['auroc'])
        print(f'[{i+1}/{n_total}] {term} AUROC={m["auroc"]:.4f} '
              f'top_prot={prot_names[int(np.argmax(sp))]} top_meta={int(np.argmax(sm))}', flush=True)

    np.savez(os.path.join(OUT_DIR, cli.out + '.npz'),
             terms=np.array(terms),
             saliency_prot=np.stack(sal_p), saliency_meta=np.stack(sal_m),
             base_auroc=np.array(auroc_all),
             prot_names=np.array(prot_names),
             meta_names=np.array(open(os.path.join(PREPROCESSED_V2_DIR, 'meta_names.txt')).read().strip().split('\n')))
    print('saved:', os.path.join(OUT_DIR, cli.out + '.npz'), flush=True)


if __name__ == '__main__':
    main()
