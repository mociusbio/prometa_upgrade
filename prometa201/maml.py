# -*- coding: utf-8 -*-
"""MAML 训练框架 — CLSDual 与 CLSCA 共用的 Stage 2 训练逻辑。

文件导览(每个函数在哪一节):
  set_seed            随机种子固定
  grad_key_filter     inner loop 更新哪些参数: 参数名含 classifier/gate/meta_fusion 的
  info_nce            对称 InfoNCE 对比损失 (附加组件, 默认关闭)
  train_step          一次 outer-loop 更新 = 逐任务 inner-adapt + query loss (+ 可选 InfoNCE)
  evaluate            support 上 inner-adapt (inner_step+5 步), query 上算指标
  compute_task_metrics  AUROC/AUPRC/F1/acc/precision/recall
  save_results        结果存 json + 训练曲线 png

训练逻辑(与原版 ProMeta 完全一致, 两套机制共用):
  inner loop: 只在 support 上更新分类头(参数名含 'classifier'/'gate'),
              每参数用可学习学习率 alphas (Meta-SGD 风格), inner_step=5 步
  outer loop: Adam 更新全部参数(包括两个 encoder, 均不冻结),
              loss = 各任务 query 上 focal loss 的均值 (+ 可选 cl_weight * InfoNCE)
"""
import os
import random
import datetime
import json
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import (f1_score, accuracy_score, precision_score, recall_score)
import torchmetrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from prometa201.layers import FocalLoss


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def grad_key_filter(k):
    """inner loop 参数过滤器: 只更新名字含 classifier/gate/meta_fusion 的参数。

    对 CLSDual/CLSCA 而言命中的是: classifier(分类头) + gate_logits(蛋白 tokenizer 门控)
    [+ mod_gate(模态门控, 启用时)]。encoder 主体不在 inner loop 更新。"""
    return ('classifier' in k) or ('gate' in k) or ('meta_fusion' in k)


def info_nce(a, b, temperature=0.07):
    """对称 InfoNCE: a[i] 与 b[i] 为正样本对 (同一病人的蛋白 CLS 与代谢 CLS),
     batch 内其余互为负样本。返回标量 loss。

    作用: 把同一病人两个组学的全局表征拉近 (不依赖 per-disease 信号,
    所以能在 few-shot 下生效 —— 见 docs/RESULTS.md 对比学习一节)。
    a, b: [N, d]。temperature τ 越小对难负样本越敏感。
    """
    n = a.size(0)
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    logits = a @ b.t() / temperature          # [N, N], 对角 = 正样本
    labels = torch.arange(n, device=a.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def train_step(model, batch, outer_optimizer, args, device):
    """一次 outer-loop 更新。batch = DualOmicsDataset 的一个 batch (多任务)。

    返回 (总loss, 纯focal loss, 全部任务query预测概率, 对应标签)。
    """
    q_prot, q_meta, q_lb, s_prot, s_meta, s_lb = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]
    s_prot, s_meta, s_lb = s_prot.to(device), s_meta.to(device), s_lb.to(device)
    q_prot, q_meta, q_lb = q_prot.to(device), q_meta.to(device), q_lb.to(device)

    inner_criterion = FocalLoss(alpha=0.5, gamma=args.focal_gamma)
    outer_criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    outer_optimizer.zero_grad()

    meta_loss_sum = 0.0; focal_loss_sum = 0.0
    batch_preds_collect = []; batch_labels_collect = []
    B_tasks = s_prot.shape[0]
    all_params = {n: p for n, p in model.named_parameters() if 'alphas' not in n}

    for i in range(B_tasks):
        # ---------- inner loop: 本任务的"临时参数" fast_weights ----------
        fast_weights = {k: v for k, v in all_params.items()}
        alphas = model.alphas
        grad_keys = [k for k in fast_weights.keys() if grad_key_filter(k)]
        t_s_prot, t_s_meta, t_s_lb = s_prot[i], s_meta[i], s_lb[i]
        t_q_prot, t_q_meta, t_q_lb = q_prot[i], q_meta[i], q_lb[i]

        for _ in range(args.inner_step):
            preds, gate_vals = model.functional_forward(t_s_prot.unsqueeze(0), t_s_meta.unsqueeze(0), fast_weights)
            preds = preds.squeeze(0)
            cls_loss = inner_criterion(preds, t_s_lb.unsqueeze(-1))
            l1_loss = args.l1_lambda * torch.norm(gate_vals, p=1)     # gate 稀疏正则
            total_inner_loss = cls_loss + l1_loss
            grads_dict = torch.autograd.grad(total_inner_loss, [fast_weights[n] for n in grad_keys],
                                             create_graph=True, allow_unused=True)
            for idx, name in enumerate(grad_keys):
                g = grads_dict[idx]
                if g is not None:
                    fast_weights[name] = fast_weights[name] - alphas[name.replace('.', '_')] * g

        # ---------- outer loss: inner-adapt 后的模型在本任务 query 上的表现 ----------
        q_logits, _ = model.functional_forward(t_q_prot.unsqueeze(0), t_q_meta.unsqueeze(0), fast_weights)
        q_logits = q_logits.squeeze(0)
        batch_preds_collect.append(torch.sigmoid(q_logits).detach().cpu())
        batch_labels_collect.append(t_q_lb.detach().cpu())
        task_loss = outer_criterion(q_logits, t_q_lb.unsqueeze(-1))
        meta_loss_sum += task_loss
        focal_loss_sum += task_loss.item()

    # ---------- 对比学习辅助损失 (默认关闭: cl_weight=0 时行为与原版完全一致) ----------
    cl_w = getattr(args, 'cl_weight', 0.0)
    cl_loss_val = 0.0
    if cl_w > 0 and hasattr(model, 'extract_cls'):
        # 把 batch 内所有 support+query 样本的双组学 CLS 拉近
        x_p = torch.cat([s_prot.reshape(-1, s_prot.size(-1)),
                         q_prot.reshape(-1, q_prot.size(-1))], dim=0)
        x_m = torch.cat([s_meta.reshape(-1, s_meta.size(-1)),
                         q_meta.reshape(-1, q_meta.size(-1))], dim=0)
        cls_p, cls_m = model.extract_cls(x_p, x_m)
        cl_loss = info_nce(cls_p, cls_m, getattr(args, 'cl_temperature', 0.07))
        cl_loss_val = cl_loss.item()
        model.last_cl_loss = cl_loss_val
        loss_for_backward = meta_loss_sum / B_tasks + cl_w * cl_loss
    else:
        loss_for_backward = meta_loss_sum / B_tasks
    loss_for_backward.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    outer_optimizer.step()
    return (loss_for_backward.item(), focal_loss_sum / B_tasks,
            torch.cat(batch_preds_collect), torch.cat(batch_labels_collect))


def compute_task_metrics(preds, labels, threshold=0.5):
    try:
        preds_t = preds.clone().detach().cpu().float().view(-1)
        labels_t = labels.clone().detach().cpu().int().view(-1)
        if len(torch.unique(labels_t)) < 2:
            return None
        auroc = torchmetrics.functional.auroc(preds_t, labels_t, task='binary')
        auprc = torchmetrics.functional.average_precision(preds_t, labels_t, task='binary')
        preds_bin = (preds_t.numpy() >= threshold).astype(int)
        labels_np = labels_t.numpy()
        return {"auroc": float(auroc), "auprc": float(auprc),
                "f1": f1_score(labels_np, preds_bin), "accuracy": accuracy_score(labels_np, preds_bin),
                "precision": float(precision_score(labels_np, preds_bin, zero_division=0)),
                "recall": float(recall_score(labels_np, preds_bin, zero_division=0))}
    except Exception as e:
        print(f"[Error in Metrics] {e}")
        return None


def evaluate(model, task_loader, args, device, mode="Valid"):
    """逐疾病: support 上 inner-adapt (inner_step+5 步, 与原版一致), query 上算指标。

    返回 (宏平均 summary, 每病 detail 列表)。
    启用 mod_gate 时, 每个疾病 detail 里会记录该疾病学到的融合权重 mod_gate_w。
    """
    model.eval()
    task_results = []
    acc_dict = {k: [] for k in ["auroc", "auprc", "f1", "accuracy"]}
    eval_criterion = FocalLoss(alpha=0.5, gamma=args.focal_gamma)
    for batch in tqdm(task_loader, desc=f"Eval ({mode})", leave=False):
        q_prot, q_meta, q_lb = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        s_prot, s_meta, s_lb = batch[3].to(device), batch[4].to(device), batch[5].to(device)
        term_name = batch[6][0]
        if len(torch.unique(s_lb[0])) < 2:
            continue
        fast_weights = {n: p.clone().detach().requires_grad_(True) for n, p in model.named_parameters() if 'alphas' not in n}
        alphas = {k: v.clone().detach() for k, v in model.alphas.items()}
        for _ in range(args.inner_step + 5):
            preds, gate_vals = model.functional_forward(s_prot, s_meta, fast_weights)
            preds = preds.squeeze(0)
            loss = eval_criterion(preds, s_lb.squeeze(0).unsqueeze(-1)) + args.l1_lambda * torch.norm(gate_vals, p=1)
            gk = [k for k in fast_weights.keys() if grad_key_filter(k)]
            target_params = [fast_weights[n] for n in gk]
            grads = torch.autograd.grad(loss, target_params, create_graph=False, allow_unused=True)
            for idx, name in enumerate(gk):
                if grads[idx] is not None:
                    fast_weights[name] = fast_weights[name] - alphas[name.replace('.', '_')] * grads[idx]
        with torch.no_grad():
            q_logits, _ = model.functional_forward(q_prot, q_meta, fast_weights)
            q_probs = torch.sigmoid(q_logits).squeeze(0)
            if torch.isnan(q_probs).any():
                continue
            if len(torch.unique(q_lb.squeeze(0))) < 2:
                continue
        try:
            metrics = compute_task_metrics(q_probs, q_lb.squeeze(0))
            if metrics:
                metrics['disease_term'] = str(term_name)
                if 'mod_gate' in fast_weights:   # 模态门控启用时导出该疾病学到的融合权重
                    metrics['mod_gate_w'] = F.softmax(fast_weights['mod_gate'].detach(), dim=0).cpu().tolist()
                task_results.append(metrics)
                for k in acc_dict:
                    if k in metrics:
                        acc_dict[k].append(metrics[k])
        except Exception:
            pass

    summary = {k: float(np.mean(v)) if len(v) > 0 else 0.0 for k, v in acc_dict.items()}
    return summary, task_results


def plot_and_save_curves(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(epochs, history['train_loss'], 'b-o'); axes[0].set_title('Training Loss')
    axes[1].plot(epochs, history['val_auroc'], 'g-o'); axes[1].set_title('Validation AUROC')
    axes[2].plot(epochs, history['val_auprc'], 'm-o'); axes[2].set_title('Validation AUPRC')
    for ax in axes: ax.grid(True)
    plt.tight_layout(); plt.savefig(save_path); plt.close()


def save_results(summary, task_results, args, model_name, extra=None, history=None):
    """测试指标存 json (含每病 detail + 配置), 训练曲线存 png。"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(args.base_path, "benchmark_results", f"support_{args.support_size}")
    os.makedirs(result_dir, exist_ok=True)
    filename_base = f"{model_name}_{getattr(args, 'init', 'x')}_seed{args.random_seed}_{timestamp}_{os.getpid()}"
    json_path = os.path.join(result_dir, f"{filename_base}.json")
    if history is not None:
        try:
            plot_and_save_curves(history, os.path.join(result_dir, f"{filename_base}.png"))
        except Exception:
            pass
    data = {"model": model_name, "experiment_name": 'ProMeta-DualOmics',
            "config": {"support_size": args.support_size, "batch_size": args.batch_size,
                       "outer_lr": args.outer_lr, "inner_lr": args.inner_lr, "inner_step": args.inner_step,
                       "dropout": args.dropout_rate, "seed": args.random_seed, "init": getattr(args, 'init', '')},
            "timestamp": timestamp, "summary_metrics": summary,
            "per_task_details": task_results}
    if extra:
        data['extra'] = extra
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Results saved: {json_path}")
