# -*- coding: utf-8 -*-
"""Stage 1: CLSCA 双分支 Cross-Attention 预训练 (相互预测 + 自重建, 无泄漏)

损失与 pretrain_cls_dual 完全相同:
  L = 4.0*MSE(p2m) + 1.0*MSE(m2p) + 0.5*MSE(recon_p) + 0.5*MSE(recon_m) + 1e-3*L1(gate)
区别只在模型 (PretrainCLSDualCA, 表征支路双边 CA / 预测支路单边可学习 query)。
val 时同时记录 max/mean pooling 两套 R2 (低成本消融记录)。

用法: python prometa201/pretrain_cls_ca.py [--epochs 50] [--gpu_id 0] [--d_model 64]
"""
import os
import sys
import time
import pickle
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prometa201.paths import (PREPROCESSED_V2_DIR, PKL_INPUT_DIR, CPDB_FILE_PATH,
                              PROT_NAMES_FILE, OUTPUT_DIR)
from prometa201.omics_encoders import load_pathway_mask
from prometa201.models_cls_ca import PretrainCLSDualCA


class Config:
    PATHWAY_FILE = CPDB_FILE_PATH
    PREPROCESSED_DIR = PREPROCESSED_V2_DIR
    PROT_NAMES_FILE = PROT_NAMES_FILE
    DATACORRECT_DIR = PKL_INPUT_DIR
    SAVE_DIR = os.path.join(OUTPUT_DIR, 'pretrain_cls_ca')
    os.makedirs(SAVE_DIR, exist_ok=True)
    LOG_FILE = os.path.join(OUTPUT_DIR, 'logs', 'pretrain_cls_ca.log')
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    META_DIM = 251
    PROT_DIM = 2923
    D_MODEL = 64
    NHEAD = 2
    NUM_CA_LAYERS = 2
    DROPOUT = 0.5
    POOL = 'max'

    BATCH_SIZE = 64
    EPOCHS = 50
    LR = 1e-4
    WEIGHT_DECAY = 1e-5
    W_P2M, W_M2P, W_RP, W_RM = 4.0, 1.0, 0.5, 0.5
    GATE_L1_W = 1e-3
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def log(msg):
    print(msg, flush=True)
    with open(Config.LOG_FILE, 'a') as f:
        f.write(msg + '\n')


def load_datacorrect_eids(dc_dir, split):
    eids = set()
    for kind in ['cases', 'controls']:
        with open(os.path.join(dc_dir, f'term2pre_{kind}_{split}.pkl'), 'rb') as f:
            d = pickle.load(f)
        for term, lst in d.items():
            eids.update(int(x) for x in lst)
    return eids


def load_features_for_eids(pre_dir, want_eids):
    X_meta = np.concatenate([np.load(os.path.join(pre_dir, 'X_meta_train.npy')),
                             np.load(os.path.join(pre_dir, 'X_meta_val.npy'))], axis=0)
    X_prot = np.concatenate([np.load(os.path.join(pre_dir, 'X_prot_train.npy')),
                             np.load(os.path.join(pre_dir, 'X_prot_val.npy'))], axis=0)
    eids = np.concatenate([np.load(os.path.join(pre_dir, 'eids_train.npy')),
                           np.load(os.path.join(pre_dir, 'eids_val.npy'))], axis=0)
    eid_to_idx = {int(e): i for i, e in enumerate(eids)}
    idx = [eid_to_idx[int(e)] for e in want_eids if int(e) in eid_to_idx]
    return X_meta[idx], X_prot[idx]


class DualDataset(Dataset):
    def __init__(self, X_meta, X_prot):
        self.X_meta = torch.from_numpy(X_meta).float()
        self.X_prot = torch.from_numpy(X_prot).float()

    def __len__(self):
        return len(self.X_meta)

    def __getitem__(self, idx):
        return self.X_prot[idx], self.X_meta[idx]


def compute_losses(out, x_meta, x_prot, c):
    """Stage 1 预训练总损失 (与 pretrain_cls_dual 相同)"""
    l_p2m = torch.nn.functional.mse_loss(out['pred_meta'], x_meta)
    l_m2p = torch.nn.functional.mse_loss(out['pred_prot'], x_prot)
    l_rp = torch.nn.functional.mse_loss(out['recon_prot'], x_prot)
    l_rm = torch.nn.functional.mse_loss(out['recon_meta'], x_meta)
    l_gate = torch.norm(out['gate'], p=1)
    total = (c.W_P2M * l_p2m + c.W_M2P * l_m2p + c.W_RP * l_rp + c.W_RM * l_rm
             + c.GATE_L1_W * l_gate)
    return {'p2m': l_p2m, 'm2p': l_m2p, 'rp': l_rp, 'rm': l_rm, 'gate': l_gate, 'total': total}


@torch.no_grad()
def validate(model, loader, c):
    model.eval()
    sums = {k: 0.0 for k in ['p2m', 'm2p', 'rp', 'rm', 'total']}
    r2 = {'max': {k: [] for k in ['p2m', 'm2p', 'rp', 'rm']},
          'mean': {k: [] for k in ['p2m', 'm2p', 'rp', 'rm']}}
    T = {k: [] for k in ['p2m', 'm2p', 'rp', 'rm']}
    for x_prot, x_meta in loader:
        x_prot, x_meta = x_prot.to(c.DEVICE), x_meta.to(c.DEVICE)
        ls_saved = None
        for mode in ['max', 'mean']:
            out = model(x_prot, x_meta, pool=mode)
            ls = compute_losses(out, x_meta, x_prot, c)
            if mode == c.POOL:
                ls_saved = ls
                for k in sums:
                    sums[k] += ls[k].item()
            r2[mode]['p2m'].append(out['pred_meta'].cpu().numpy())
            r2[mode]['m2p'].append(out['pred_prot'].cpu().numpy())
            r2[mode]['rp'].append(out['recon_prot'].cpu().numpy())
            r2[mode]['rm'].append(out['recon_meta'].cpu().numpy())
        T['p2m'].append(x_meta.cpu().numpy()); T['m2p'].append(x_prot.cpu().numpy())
        T['rp'].append(x_prot.cpu().numpy()); T['rm'].append(x_meta.cpu().numpy())
    for mode in ['max', 'mean']:
        for k in r2[mode]:
            r2[mode][k] = r2_score(np.concatenate(T[k]).flatten(), np.concatenate(r2[mode][k]).flatten())
    n = len(loader)
    return {k: v / n for k, v in sums.items()}, r2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=Config.EPOCHS)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--d_model', type=int, default=Config.D_MODEL)
    cli = parser.parse_args()

    c = Config()
    c.D_MODEL = cli.d_model
    torch.cuda.set_device(cli.gpu_id) if torch.cuda.is_available() else None
    log(f"Device: {c.DEVICE}, pool={c.POOL}, d_model={c.D_MODEL}")
    with open(c.PROT_NAMES_FILE) as f:
        prot_names = f.read().strip().split('\n')
    pathway_mask, unknown_indices = load_pathway_mask(c.PATHWAY_FILE, prot_names)

    train_eids = load_datacorrect_eids(c.DATACORRECT_DIR, 'train')
    val_eids = load_datacorrect_eids(c.DATACORRECT_DIR, 'valid')
    log(f"data_correct train patients: {len(train_eids)}, valid: {len(val_eids)}")
    Xm_tr, Xp_tr = load_features_for_eids(c.PREPROCESSED_DIR, train_eids)
    Xm_va, Xp_va = load_features_for_eids(c.PREPROCESSED_DIR, val_eids)
    log(f"Train: meta{Xm_tr.shape} protXp{Xp_tr.shape} | Val: meta{Xm_va.shape}")
    np.nan_to_num(Xm_tr, copy=False); np.nan_to_num(Xp_tr, copy=False)
    np.nan_to_num(Xm_va, copy=False); np.nan_to_num(Xp_va, copy=False)

    train_loader = DataLoader(DualDataset(Xm_tr, Xp_tr), batch_size=c.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(DualDataset(Xm_va, Xp_va), batch_size=c.BATCH_SIZE, shuffle=False)

    model = PretrainCLSDualCA(meta_dim=c.META_DIM, prot_dim=c.PROT_DIM, d_model=c.D_MODEL,
                              nhead=c.NHEAD, num_ca_layers=c.NUM_CA_LAYERS, dropout=c.DROPOUT,
                              pathway_mask=pathway_mask, unknown_indices=unknown_indices,
                              pool=c.POOL).to(c.DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"PretrainCLSDualCA params: {n_params/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=c.LR, weight_decay=c.WEIGHT_DECAY)
    best_val, best_epoch = float('inf'), 0
    for epoch in range(1, cli.epochs + 1):
        start = time.time()
        model.train()
        tl = 0.0
        for x_prot, x_meta in train_loader:
            x_prot, x_meta = x_prot.to(c.DEVICE), x_meta.to(c.DEVICE)
            optimizer.zero_grad()
            out = model(x_prot, x_meta)
            losses = compute_losses(out, x_meta, x_prot, c)
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tl += losses['total'].item()
        va, r2 = validate(model, val_loader, c)
        m, mn = r2['max'], r2['mean']
        log(f"Epoch {epoch}/{cli.epochs} | Train {tl/len(train_loader):.4f} | "
            f"Val total {va['total']:.4f} (p2m{va['p2m']:.3f} m2p{va['m2p']:.3f} rp{va['rp']:.3f} rm{va['rm']:.3f}) | "
            f"R2[max]: p2m={m['p2m']:.4f} m2p={m['m2p']:.4f} recon_p={m['rp']:.4f} recon_m={m['rm']:.4f} | "
            f"R2[mean]: p2m={mn['p2m']:.4f} m2p={mn['m2p']:.4f} | "
            f"{time.time()-start:.1f}s")
        if va['total'] < best_val:
            best_val, best_epoch = va['total'], epoch
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'val_loss': best_val, 'r2_max': m, 'r2_mean': mn,
                        'pathway_mask': pathway_mask, 'unknown_indices': unknown_indices},
                       os.path.join(c.SAVE_DIR, 'best_model.pt'))
            log(f"  -> Saved best (val={best_val:.4f})")
    log(f"Done. Best epoch {best_epoch}, val {best_val:.4f}")


if __name__ == '__main__':
    main()
