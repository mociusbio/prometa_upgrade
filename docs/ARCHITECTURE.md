# 架构与代码导览（小白版）

> 目标：读完这篇，你能说清楚每一层数据怎么流动、每个 loss 在代码哪一行、
> MAML 的 inner/outer loop 到底在更新谁。

## 0. 一句话总览

```
病人的两种组学数据 (2923蛋白 + 251代谢物)
   │  Stage 1 预训练 (没有疾病标签，纯数据): 互相预测 + 自重建 → 学到"两种组学怎么互相表达"
   ▼
双分支 encoder (蛋白一支 / 代谢一支)
   │  Stage 2 MAML (有 781 个疾病的标签): 每个疾病给 32 个样本快速适应，预测没见过的病人
   ▼
每个疾病的诊断概率 (91 个 held-out 疾病上评测)
```

两个 Stage 用**同一套 encoder 权重**（Stage 1 训完把 encoder 搬给 Stage 2 热启动，预测头丢弃）。

## 1. 数据怎么进模型（`dataset.py` + `omics_encoders.py`）

### 1.1 一个"任务"是什么（`dataset.py: DualOmicsDataset`）

MAML 把**每个疾病当成一个独立的二分类任务**（病例=1，对照=0）。一个样本返回 7 样东西：

```
qX_prot, qX_meta, qY     ← query 集: 考卷 (≤128 人, 算 loss 和指标)
sX_prot, sX_meta, sY     ← support 集: 教材 (32 人 = 16 病例 + 16 对照, 供 inner loop 临时更新)
term                     ← 疾病名 (如 ST19_INJURI_HEAD)
```

support/query 的切分：每个疾病的病例/对照列表用固定种子 shuffle 后，前 32 个留作 support 池，
剩下作 query 池，互不重叠（`dataset.py:120-141`）。

### 1.2 蛋白怎么变成 token（`omics_encoders.py: PathwayGatedTokenizer`）

2923 个蛋白太多、太噪，原版 ProMeta 的解法是**按 KEGG 通路聚合成 token**：

```
x_prot [B, 2923]
  │  Linear(1→32) + 位置编码            # 每个蛋白的表达值变成一个 32 维小向量
  │  × sigmoid(gate_logits)             # ★ 每个蛋白一个可学习的门 [0,1]，相当于"蛋白重要性"
  ▼
 × pathway_mask [327条通路 × 2923蛋白]   # einsum 加权求和：同一条通路的蛋白加起来
  │  Linear(32→64)
  ▼
327 个"通路 token"
  │  cat[ 不属于任何通路的 1309 个蛋白 → MLP → 32 个 "unknown token" ]
  ▼
[B, 359, 64] 蛋白 token 序列
```

- `gate_logits`（代码 `omics_encoders.py:60-62`）就是 RQ2 分析里"蛋白重要性"的来源——
  但注意它是**全局参数**，所有疾病共享（后来发现它逐病完全相同，见 RQ2 报告）。
- 掩码矩阵由 `load_pathway_mask` 从 CPDB 的 KEGG 注释构建（`omics_encoders.py` 末尾）。

### 1.3 代谢怎么变成 token（`omics_encoders.py: SimpleMetaEncoder`）

代谢物没有通路注释可用，用最简单的办法：**一个代谢物一个 token**：

```
x_meta [B, 251] → Linear(1→64) + 位置编码 → [B, 251, 64]
```

## 2. 机制一 CLSDual（`models_cls_dual.py`）：两支 transformer，CLS 拼接

```
                      Stage 1 预训练 (PretrainCLSDual)                  Stage 2 分类 (CLSDualMamlModel)
                      ─────────────────────────────                      ─────────────────────────────

蛋白: token[359] ─ +cls_token → ×2 transformer层 → 取第0位 ─── cls_prot[64]
代谢: token[251] ─ +meta_cls_token → ×2 transformer层(独立参数) → 取第0位 ─── cls_meta[64]

                                                            ┌ shortcut_proj(x_prot)[64]  ← 抄近路
      头1 head_p2m: 蛋白token矩阵 → 预测代谢谱 [251]         │
      头2 head_m2p: 代谢token矩阵 → 预测蛋白谱 [2923]  ──丢──▶│ z = concat(cls_prot + shortcut, cls_meta) [128]
      头3 head_recon_p: cls_prot → 重建蛋白谱 [2923]   ──弃──▶│ logits = Linear(128→1)(z)
      头4 head_recon_m: cls_meta → 重建代谢谱 [251]          │
                                                            ▼
                                                      疾病诊断概率
```

- 两个分支各两层 `MetaTransformerLayer`（自注意力，`layers.py`），**参数完全独立**。
- "取第 0 位"就是 CLS 机制：序列最前面塞一个可学习向量，注意力过后它变成全局摘要。
- `shortcut_proj`（`models_cls_dual.py:169`）：原始蛋白谱直接线性压到 64 维加到 cls_prot 上，
  是原版 ProMeta 的防信息丢失设计。

## 3. 机制二 CLSCA（`models_cls_ca.py`）：Cross-Attention 互相"看"，再池化

CLSDual 的两个分支**全程互不通信**，直到最后拼接。CLSCA 让它们在 token 级别就互相看：

```
              Stage 1 (PretrainCLSDualCA)                        Stage 2 (CLSDualCAMamlModel)
              ───────────────────────────                        ────────────────────────────

  toks_p [359,64] ←── tokenizer(x_prot)
  toks_m [251,64] ←── encoder(x_meta)

  ★表征支路 (双边, 进下游):                          只用表征支路:
    M_repr = CA(Q=toks_m, KV=toks_p)   代谢被蛋白"告知"
    P_repr = CA(Q=toks_p, KV=toks_m)   蛋白被代谢"告知"     z = concat(maxpool(P_repr)+shortcut,
    z_m = maxpool(M_repr); z_p = maxpool(P_repr)                  maxpool(M_repr)) → Linear(128→1)

  预测支路 (单边, 训完丢弃):
    p2m: q=可学习query[251], KV=toks_p(只有蛋白) → CA → 读出头 → 预测 x_meta
    m2p: q=可学习query[359], KV=toks_m(只有代谢) → CA → 映射头 → 预测 x_prot
    (两套支路共享 ca_layers 权重 —— 预测支路逼 CA 层学跨组学映射,
     表征支路拿到被对侧"告知"过的表示)
```

**为什么预测支路要"单边"**（这是本项目最重要的设计决策）：
如果预测代谢谱时让代谢自己的 token 参与（双边），模型直接抄答案，R² 虚高到 0.996 但什么都没学到
（本项目早期 BCA 版本的教训，也是 scMoFormer 等文献指出的泄漏）。所以预测时 Q 用**可学习参数**、
KV 只给**源模态**——训练时只有单边信息，逼模型真学跨组学映射；预测（下游）时两边都有。
类比 BERT：训练时 mask、使用时完整输入。

`CABlock`（`models_cls_ca.py:30`）就是一个手写交叉注意力块：Q 来自 query、K/V 来自 kv，
残差 + LayerNorm + GELU FFN，支持 `functional_forward` 供 MAML 传临时参数。

## 4. Stage 1 的 loss 在哪里（两个预训练脚本里的 `compute_losses`）

`pretrain_cls_dual.py:111` 和 `pretrain_cls_ca.py`（相同公式）：

```python
total = 4.0*MSE(pred_meta, x_meta)      # p2m 相互预测: 蛋白 → 代谢谱
      + 1.0*MSE(pred_prot, x_prot)      # m2p 相互预测: 代谢 → 蛋白谱
      + 0.5*MSE(recon_prot, x_prot)     # 蛋白自重建
      + 0.5*MSE(recon_meta, x_meta)     # 代谢自重建
      + 1e-3*L1(gate)                   # 蛋白门控稀疏正则
```

- p2m 权重给 4 倍是因为代谢谱更好预测、需要放大梯度（不对称的两个任务）。
- 评价预训练好坏用 R²（脚本 log 里 `R2: p2m/m2p/recon_p/recon_m`）。

## 5. Stage 2 的 MAML 到底在干嘛（`maml.py`）

### 5.1 inner loop vs outer loop（`maml.py: train_step`）

```
对一个 batch 里的每个疾病任务:
  fast_weights = 复制一份当前参数
  重复 inner_step=5 次:
      在 support 上算 loss = FocalLoss(预测, 标签) + l1*||gate||
      对 grad_key_filter 选中的参数求梯度        ← 只选名字含 classifier/gate 的!
      fast_weights -= alphas * grad              ← alphas 是每个参数的可学习学习率 (Meta-SGD)
  在 query 上用 fast_weights 算 focal loss       ← outer loss

所有任务的 outer loss 取平均 (+ 可选 InfoNCE), backward, Adam 更新全部参数
```

- **inner loop 只更新分类头**（classifier + 蛋白 gate_logits [+ mod_gate]），
  模拟"拿到 32 个样本后快速微调诊断头"；encoder 不动。
- **outer loop 更新全部参数**（包括两个 encoder），目标是"微调后的头在新病人上表现好"——
  所以 encoder 学的是"如何初始化才能让 32 个样本就够调出一个好分类器"。
- 评测时（`evaluate`，`maml.py:261`）逻辑相同但 inner 跑 `inner_step+5=10` 步、不建二阶梯度。

### 5.2 两个附加组件

**InfoNCE 对比学习**（`maml.py: info_nce` + `train_step` 末尾，开关 `--cl_weight`）：

```
把 batch 里所有病人的 (cls_prot, cls_meta) 成对拿出来
loss = 0.5 * [ CE(cls_prot·cls_meta^T/τ, 同病人) + CE(转置, 同病人) ]
```

同一病人的两个组学 CLS 是正样本，其他病人互为负样本。它**不依赖 per-disease 信号**
（全局约束），所以能在 few-shot 下生效——这是它唯一跑赢的门控类方法的原因。
实验最优：`--cl_weight 0.1 --cl_temperature 0.07`，+0.003 AUROC。

**可学习模态门控 mod_gate**（`models_cls_dual.py:130` / `models_cls_ca.py:175`，开关 `--mod_gate 1`）：

```
z = concat(w_prot * cls_prot,  w_meta * cls_meta),  (w_prot, w_meta) = softmax(mod_gate)
```

mod_gate 名字含 'gate' → inner loop 会按疾病适配，初衷是"每个疾病自己决定信哪个组学"。
**实验结果：塌缩**——每个 seed 内 91 个病的权重跨病 std≈0.0001，从没分化过；
不同 seed 塌到不同常数（0.499~0.565）。64 个 support 样本里"哪个组学更重要"的信号不可识别。

### 5.3 其他 loss 位置速查

| Loss | 位置 |
|---|---|
| Stage 1 预训练总损失 | `pretrain_cls_dual.py: compute_losses`（CA 版同公式） |
| inner loop 分类损失 | `maml.py: train_step`，`FocalLoss(alpha=0.5)` |
| outer loop 分类损失 | `maml.py: train_step`，`FocalLoss(alpha=0.75)` |
| gate L1 正则 | `maml.py: train_step`，`l1_lambda * ||gate||`（inner 和 evaluate 都有） |
| InfoNCE | `maml.py: info_nce` 函数 + `train_step` 末尾（默认关闭） |
| FocalLoss 本体 | `layers.py: FocalLoss` |

## 6. RQ2 分析脚本（`analysis/`）在算什么

| 脚本 | 干什么 | 关键产物 |
|---|---|---|
| `extract_rq2.py` | 每病 inner-adapt 后导蛋白 gate + 代谢物置换重要性 | gate_w, perm_drop |
| `extract_rq2_saliency.py` | 每病算 query 正类 logit 对原始输入的 \|梯度\| | saliency_prot/meta [91, D] |
| `meta_univariate_auc.py` | 每病 × 每代谢物的单变量 AUROC（与模型无关） | auc [91, 251] |

三个脚本的分析结论（阴性也是结论）：
1. 融合模型的分类器**实际忽略代谢支**（遮挡实验：251 个代谢物全置零 AUROC 变化 <0.01）；
2. tokenizer gate 是全局参数，**不能逐病比较**；
3. saliency（减跨病均值后）恢复教科书生物学：甲减→TSHB(#4)、呼吸困难→NT-proBNP(#1)。

## 7. 改代码时最容易踩的坑

- **`functional_forward` 无处不在**：MAML 要求每层都能用外部传入的 params 计算，
  所以所有层只有 functional 版本。新增层时必须照 `MetaTransformerLayer` /
  `CABlock` 的写法实现 `functional_forward(x, params, prefix)`。
- **inner loop 参数靠名字过滤**（`grad_key_filter`）：想让某个参数被 inner loop 更新，
  名字里必须含 `classifier`/`gate`/`meta_fusion`——`mod_gate` 就是利用这一点挂上 inner loop 的。
- **Stage 1/Stage 2 权重名必须一致**：热启动靠 `load_state_dict(strict=False)` 按名字对，
  改名会破坏热启动（`load_pretrained_encoders` 只跳过 `head_*` 和 queries）。
- **每个训练任务 ~15GB 显存**：一卡一任务，多卡并行见 `scripts/run_3seeds_example.sh`。
