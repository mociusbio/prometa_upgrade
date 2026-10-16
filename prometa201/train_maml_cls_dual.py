# -*- coding: utf-8 -*-
"""Stage 2: CLSDual 双分支 MAML 疾病分类 (原版 ProMeta 训练逻辑)

模型: CLSDualMamlModel = 双分支 encoder(可热启动) + shortcut + concat + Linear(128->1)
inner loop: 只更新参数名含 'classifier'/'gate' 的 (classifier + prot gate_logits [+ mod_gate])
outer loop: Adam 更新全部参数 (双 encoder 均不冻结), Meta-SGD alphas

关键开关:
  --init           Stage 1 预训练 ckpt 路径; 'cls' = outputs/pretrain_cls_dual/best_model.pt;
                   'none' = 随机初始化; 其他 = 视为 ckpt 路径
  --cl_weight      InfoNCE 对比学习权重 (0=关闭, 默认; 推荐 0.1, 配 --cl_temperature 0.07)
  --mod_gate       1=启用可学习模态门控 (实验结论: 塌缩为全局常数, 不推荐)

用法示例:
  # Stage 1 -> Stage 2 (预训练热启动, 3 seeds 各跑一次)
  python prometa201/train_maml_cls_dual.py --gpu_id 0 --init cls --random_seed 42
  # 加对比学习 (最优配置):
  python prometa201/train_maml_cls_dual.py --gpu_id 0 --init cls --cl_weight 0.1 --cl_temperature 0.07
"""
import os
import sys
import argparse
import datetime
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prometa201.paths import (PREPROCESSED_V2_DIR, PKL_INPUT_DIR, CPDB_FILE_PATH,
                              PROT_NAMES_FILE, OUTPUT_DIR)
from prometa201.omics_encoders import load_pathway_mask
from prometa201.models_cls_dual import CLSDualMamlModel, load_pretrained_encoders
from prometa201.dataset import DualOmicsDataset
from prometa201.maml import (train_step, evaluate, save_results,
                             compute_task_metrics, set_seed)

BASE_PROJECT_PATH = os.path.join(OUTPUT_DIR, 'cls_dual')
PRETRAIN_CKPT = os.path.join(OUTPUT_DIR, 'pretrain_cls_dual', 'best_model.pt')


class Args:
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
    base_path = ""
    init = "none"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--support_size', type=int, default=32)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--outer_lr', type=float, default=1e-4)
    parser.add_argument('--inner_lr', type=float, default=0.005)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--l1_lambda', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--init', type=str, default='none',
                        help="Stage 1 预训练 ckpt 路径 (只加载双分支 encoder); "
                             "'cls' = outputs/pretrain_cls_dual/best_model.pt; 'none' = 随机初始化")
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--cl_weight', type=float, default=0.0,
                        help="对比学习 loss 权重 (0=关闭, 行为与原版一致)")
    parser.add_argument('--cl_temperature', type=float, default=0.07,
                        help="InfoNCE 温度 tau")
    parser.add_argument('--mod_gate', type=int, default=0,
                        help="1=启用可学习模态门控 (inner loop 按疾病适配融合权重)")
    cli = parser.parse_args()

    args = Args()
    args.embed_dim = cli.embed_dim
    args.support_size = cli.support_size; args.batch_size = cli.batch_size
    args.outer_lr = cli.outer_lr; args.inner_lr = cli.inner_lr
    args.dropout_rate = cli.dropout; args.l1_lambda = cli.l1_lambda
    args.patience = cli.patience; args.epochs = cli.epochs
    args.random_seed = cli.random_seed
    args.cl_weight = cli.cl_weight
    args.cl_temperature = cli.cl_temperature
    args.use_mod_gate = bool(cli.mod_gate)
    args.base_path = BASE_PROJECT_PATH

    # --init: 'cls' = 默认预训练 ckpt; 'none' = 随机; 其他 = 视为 ckpt 路径
    init_path = ''
    if cli.init == 'cls':
        init_path = PRETRAIN_CKPT
    elif cli.init not in ('', 'none'):
        init_path = cli.init
    args.init = 'pre' if init_path else 'scratch'

    set_seed(cli.random_seed)
    device = torch.device(f"cuda:{cli.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Params: init={init_path or 'none'} OutLR={args.outer_lr} InLR={args.inner_lr} "
          f"Drop={args.dropout_rate} L1={args.l1_lambda} patience={args.patience}", flush=True)

    # ---- 特征 (preprocessed_v2, 按 EID 对齐蛋白/代谢) ----
    eids = np.concatenate([np.load(os.path.join(PREPROCESSED_V2_DIR, 'eids_train.npy')),
                           np.load(os.path.join(PREPROCESSED_V2_DIR, 'eids_val.npy'))], axis=0)
    X_prot = np.concatenate([np.load(os.path.join(PREPROCESSED_V2_DIR, 'X_prot_train.npy')),
                             np.load(os.path.join(PREPROCESSED_V2_DIR, 'X_prot_val.npy'))], axis=0).astype(np.float32)
    X_meta = np.concatenate([np.load(os.path.join(PREPROCESSED_V2_DIR, 'X_meta_train.npy')),
                             np.load(os.path.join(PREPROCESSED_V2_DIR, 'X_meta_val.npy'))], axis=0).astype(np.float32)
    X_prot = np.nan_to_num(X_prot); X_meta = np.nan_to_num(X_meta)
    eid_to_idx = {str(int(e)): i for i, e in enumerate(eids)}
    print(f"Patients: {len(eid_to_idx)}, prot{X_prot.shape}, meta{X_meta.shape}", flush=True)

    with open(os.path.join(PKL_INPUT_DIR, 'term2pre_cases_train.pkl'), 'rb') as f: train_case = pickle.load(f)
    with open(os.path.join(PKL_INPUT_DIR, 'term2pre_controls_train.pkl'), 'rb') as f: train_ctrl = pickle.load(f)
    with open(os.path.join(PKL_INPUT_DIR, 'term2pre_cases_valid.pkl'), 'rb') as f: valid_case = pickle.load(f)
    with open(os.path.join(PKL_INPUT_DIR, 'term2pre_controls_valid.pkl'), 'rb') as f: valid_ctrl = pickle.load(f)
    with open(os.path.join(PKL_INPUT_DIR, 'term2pre_cases_test.pkl'), 'rb') as f: test_case = pickle.load(f)
    with open(os.path.join(PKL_INPUT_DIR, 'term2pre_controls_test.pkl'), 'rb') as f: test_ctrl = pickle.load(f)

    with open(PROT_NAMES_FILE) as f:
        prot_names = f.read().strip().split('\n')
    pathway_mask, unknown_indices = load_pathway_mask(CPDB_FILE_PATH, prot_names)
    pathway_mask = pathway_mask.to(device)

    train_dataset = DualOmicsDataset(X_prot, X_meta, train_case, train_ctrl, eid_to_idx,
                                     support_size=args.support_size, mode='train', random_seed=args.random_seed)
    valid_dataset = DualOmicsDataset(X_prot, X_meta, valid_case, valid_ctrl, eid_to_idx,
                                     support_size=args.support_size, mode='valid', random_seed=args.random_seed)
    test_dataset = DualOmicsDataset(X_prot, X_meta, test_case, test_ctrl, eid_to_idx,
                                    support_size=args.support_size, mode='test', random_seed=args.random_seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = CLSDualMamlModel(num_proteins=X_prot.shape[1], num_metabolites=X_meta.shape[1],
                             args=args, pathway_mask=pathway_mask,
                             unknown_indices=unknown_indices).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"CLSDualMamlModel params: {n_params/1e6:.2f}M", flush=True)

    if init_path:
        if not os.path.exists(init_path):
            raise FileNotFoundError(f"init ckpt not found: {init_path}")
        load_pretrained_encoders(model, init_path)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.outer_lr)
    best_auroc, patience_counter = 0.0, 0
    history = {'train_loss': [], 'val_auroc': [], 'val_auprc': []}
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    best_model_path = os.path.join(BASE_PROJECT_PATH, 'benchmark_results', f"support_{args.support_size}",
                                   f"CLSDual_{args.init}_best_seed{args.random_seed}_{timestamp}_{os.getpid()}.pth")
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        l_sum, focal_sum = 0.0, 0.0
        preds_all, labels_all = [], []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            loss, focal_val, batch_probs, batch_labels = train_step(model, batch, optimizer, args, device)
            l_sum += loss; focal_sum += focal_val
            preds_all.append(batch_probs); labels_all.append(batch_labels)
        avg_loss = l_sum / len(train_loader)
        train_metrics = compute_task_metrics(torch.cat(preds_all), torch.cat(labels_all))
        if train_metrics is None:
            train_metrics = {"auroc": 0.0}
        val_summary, _ = evaluate(model, valid_loader, args, device)
        history['train_loss'].append(avg_loss)
        history['val_auroc'].append(val_summary['auroc'])
        history['val_auprc'].append(val_summary['auprc'])
        cl_info = f" | CL {getattr(model, 'last_cl_loss', 0.0):.4f}" if args.cl_weight > 0 else ""
        print(f"Epoch {epoch+1} | Loss {avg_loss:.4f} | TRAIN AUC {train_metrics['auroc']:.4f} | "
              f"VAL AUC {val_summary['auroc']:.4f}{cl_info}", flush=True)
        if val_summary['auroc'] > best_auroc:
            best_auroc = val_summary['auroc']; patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New Best (VAL AUC {best_auroc:.4f})", flush=True)
        else:
            patience_counter += 1
            print(f"No improvement {patience_counter}/{args.patience}", flush=True)
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1}", flush=True)
            break

    print("--- Final Testing ---", flush=True)
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=False))
    test_summary, test_task_results = evaluate(model, test_loader, args, device, mode="Test")
    print(f"Final Test AUROC: {test_summary['auroc']:.4f}", flush=True)
    save_results(summary=test_summary, task_results=test_task_results, args=args,
                 model_name="CLSDual", extra={'init_ckpt': init_path,
                                              'cl_weight': args.cl_weight,
                                              'cl_temperature': getattr(args, 'cl_temperature', 0.07),
                                              'mod_gate': args.use_mod_gate}, history=history)


if __name__ == '__main__':
    main()
