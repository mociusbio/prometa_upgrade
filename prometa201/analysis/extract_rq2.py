# -*- coding: utf-8 -*-
"""RQ2 特征重要性提取 (a) 蛋白 gate 排序 (b) 代谢物置换重要性 —— 91 个 test 疾病。

用法:
  python prometa201/analysis/extract_rq2.py --ckpt <best.pth> --gpu_id 0 --random_seed 42 \
      --n_perm_repeats 3 --out rq2_features_seed42

输出 (npz):
  gate_w     [91, 2923]  sigmoid(adapted tokenizer.gate_logits)  — 模型自带的蛋白重要性
  base_auroc [91]         每个疾病的 query AUROC
  perm_drop  [91, 251]    代谢物 j 置乱后 AUROC 下降 (置换重要性)
  terms / prot_names / meta_names

注意 (本研究的阴性发现): tokenizer gate 是全局参数, 91 病向量完全相同,
不能用于逐病比较; 置换重要性在融合模型上趋零 (分类器实际忽略代谢支)。
逐病分析请用 extract_rq2_saliency.py (输入梯度) 和 meta_univariate_auc.py (单变量)。
"""
import os
import sys
import argparse
import pickle
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from prometa201.paths import (PREPROCESSED_V2_DIR, PKL_INPUT_DIR, CPDB_FILE_PATH,
                              PROT_NAMES_FILE, OUTPUT_DIR)
from prometa201.omics_encoders import load_pathway_mask
from prometa201.models_cls_dual import CLSDualMamlModel
from prometa201.dataset import DualOmicsDataset
from prometa201.maml import grad_key_filter, compute_task_metrics
from prometa201.layers import FocalLoss

OUT_DIR = os.path.join(OUTPUT_DIR, 'analysis')
os.makedirs(OUT_DIR, exist_ok=True)


class Args:
    """与 train_maml_cls_dual 的评测配置保持一致 (inner adapt 10步, l1=1e-3)"""
    random_seed = 42
    query_size = 128
    batch_size = 2
    inner_lr = 0.005
    outer_lr = 1e-4
    inner_step = 5
    epochs = 100
    patience = 30
    dropout_rate = 0.5
    embed_dim = 64
    num_heads = 2
    num_layers = 2
    focal_alpha = 0.75
    focal_gamma = 2.0
    l1_lambda = 1e-3
    support_size = 32
    use_mod_gate = False


def adapt(model, s_prot, s_meta, s_lb, args, device):
    """与 evaluate() 一致的 inner adaptation (inner_step+5=10步), 返回 fast_weights"""
    eval_criterion = FocalLoss(alpha=0.5, gamma=args.focal_gamma)
    fast_weights = {n: p.clone().detach().requires_grad_(True)
                    for n, p in model.named_parameters() if 'alphas' not in n}
    alphas = {k: v.clone().detach() for k, v in model.alphas.items()}
    for _ in range(args.inner_step + 5):
        preds, gate_vals = model.functional_forward(s_prot, s_meta, fast_weights)
        preds = preds.squeeze(0)
        loss = eval_criterion(preds, s_lb.squeeze(0).unsqueeze(-1)) + args.l1_lambda * torch.norm(gate_vals, p=1)
        gk = [k for k in fast_weights.keys() if grad_key_filter(k)]
        grads = torch.autograd.grad(loss, [fast_weights[n] for n in gk],
                                    create_graph=False, allow_unused=True)
        for idx, name in enumerate(gk):
            if grads[idx] is not None:
                fast_weights[name] = fast_weights[name] - alphas[name.replace('.', '_')] * grads[idx]
    return fast_weights


@torch.no_grad()
def query_probs(model, q_prot, q_meta, fast_weights):
    logits, _ = model.functional_forward(q_prot, q_meta, fast_weights)
    return torch.sigmoid(logits).squeeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--gpu_id', type=int, default=0)
    ap.add_argument('--random_seed', type=int, default=42)
    ap.add_argument('--n_perm_repeats', type=int, default=3)
    ap.add_argument('--max_tasks', type=int, default=0, help=">0 时只跑前 N 个疾病 (冒烟测试)")
    ap.add_argument('--out', default='rq2_features_seed42')
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

    n_meta = X_meta.shape[1]
    terms, gate_all, auroc_all, perm_all = [], [], [], []
    n_total = min(cli.max_tasks, len(test_loader)) if cli.max_tasks > 0 else len(test_loader)
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
            probs = query_probs(model, q_prot, q_meta, fast_weights)
            if torch.isnan(probs).any() or len(torch.unique(q_lb[0])) < 2:
                continue
            m = compute_task_metrics(probs, q_lb.squeeze(0))
            if m is None:
                continue
            gate_w = torch.sigmoid(fast_weights['tokenizer.gate_logits'].detach()).cpu().numpy()
            base_auroc = m['auroc']
            # 置换重要性
            drops = np.zeros(n_meta, dtype=np.float32)
            q_meta0 = q_meta.clone()
            for j in range(n_meta):
                auc_sh = []
                for _ in range(cli.n_perm_repeats):
                    qm = q_meta0.clone()
                    qm[0, :, j] = qm[0, torch.randperm(qm.shape[1], device=device), j]
                    pr = query_probs(model, q_prot, qm, fast_weights)
                    mj = compute_task_metrics(pr, q_lb.squeeze(0))
                    if mj is not None:
                        auc_sh.append(mj['auroc'])
                drops[j] = base_auroc - float(np.mean(auc_sh))
        terms.append(str(term)); gate_all.append(gate_w); auroc_all.append(base_auroc); perm_all.append(drops)
        print(f'[{i+1}/{n_total}] {term} AUROC={base_auroc:.4f} '
              f'perm_max_drop={drops.max():.4f} top_meta={int(np.argmax(drops))}', flush=True)

    np.savez(os.path.join(OUT_DIR, cli.out + '.npz'),
             terms=np.array(terms), gate_w=np.stack(gate_all),
             base_auroc=np.array(auroc_all), perm_drop=np.stack(perm_all),
             prot_names=np.array(prot_names),
             meta_names=np.array(open(os.path.join(PREPROCESSED_V2_DIR, 'meta_names.txt')).read().strip().split('\n')))
    print('saved:', os.path.join(OUT_DIR, cli.out + '.npz'), flush=True)


if __name__ == '__main__':
    main()
