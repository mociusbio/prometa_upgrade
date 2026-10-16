# ProMeta 2.0.1 — 双组学 Few-Shot 疾病预测的机制研究与基线

> ## ⏱ 30 秒快速定位（给合作者/审稿人）
> - **找代码**：全部在 [`prometa201/`](prometa201/)——从 [`prometa201/maml.py`](prometa201/maml.py)
>   （MAML 训练逻辑）和 [`prometa201/models_cls_dual.py`](prometa201/models_cls_dual.py) /
>   [`models_cls_ca.py`](prometa201/models_cls_ca.py)（两套模型）开始；逐层讲解见
>   **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**
> - **找结果数字**：[docs/RESULTS.md](docs/RESULTS.md)（结论汇总）；
>   每个数字背后的原始运行记录在 [`results/runs/`](results/runs/)（68 个 JSON +
>   [MANIFEST.md](results/runs/MANIFEST.md) 清单）；逐病明细在
>   [`results/results_per_disease_ALL_methods.md`](results/results_per_disease_ALL_methods.md)
> - **找机制发现**（融合模型为何不用代谢支）：[docs/RQ2_REPORT.md](docs/RQ2_REPORT.md)
> - 跑代码的顺序：README 下方「快速开始」四行命令

---

在 ProMeta（蛋白组 few-shot 疾病诊断， pathway-tokenizer + MAML）基础上加入**代谢组**信息，
系统回答一个问题：**加入第二个组学，能不能提升、在哪些疾病上提升、以及为什么大多时候不能？**

本仓库包含完整的研究代码与结果：
- **两套融合机制**：`CLSDual`（双分支 transformer + CLS 拼接）与 `CLSCA`（双分支 Cross-Attention + token 池化）
- **两个附加组件**：InfoNCE 对比学习（小而稳定的正收益）、可学习模态门控（阴性，塌缩为全局常数）
- **完整的机制分析（RQ2）**：遮挡实验、置换重要性、输入梯度 saliency、代谢物单变量图谱 ——
  发现融合模型的决策实际完全绕开代谢支，而代谢组单独携带真实疾病信号

> 数据：UK Biobank（Olink 蛋白组 2923 + Nightingale 代谢组 251），91 个 held-out 疾病，
> 每个疾病 32 个 support 样本的小样本诊断任务。数据受 UKB 许可限制**不能公开**，
> 代码中通过 `PROMETA_DATA_ROOT` 环境变量定位数据（见「数据准备」）。

## 结果一览（91 test 疾病宏平均 AUROC，3 seeds）

| 模型 | AUROC | 备注 |
|---|---|---|
| 蛋白-only（原版 ProMeta 复现） | 0.7071 ± 0.0033 | 基线 |
| CLSDual（无预训练） | 0.7027 ± 0.0063 | 预训练贡献 +0.008 (p<1e-6) |
| **CLSDual（预训练热启动）** | **0.7103 ± 0.0041** | transformer 版最优 |
| CLSCA（预训练热启动） | 0.7039 ± 0.0032 | CA 版预训练帮倒忙 (−0.009, p<1e-5) |
| **CLSCA（无预训练）** | **0.7128 ± 0.0043** | 全局最优（与 CLSDual-pre 打平, p=0.23） |
| **CLSDual + InfoNCE (λ=0.1, τ=0.07)** | **0.7129** | 名义全局最优，配对 +0.003 |
| CLSDual/CLSCA + mod_gate | 0.7079 / 0.7084 | 阴性：门控塌缩为全局常数，从未逐病分化 |
| CLSCA + InfoNCE + mod_gate | 0.7000 | 阴性：门控与对比学习叠加有害 |

核心机制发现（详见 [docs/RESULTS.md](docs/RESULTS.md) 与 [docs/RQ2_REPORT.md](docs/RQ2_REPORT.md)）：
- 两架构的 per-disease 分类器实际**忽略代谢支**（代谢置零 AUROC 不变，蛋白置零崩到 0.5）
- 但代谢组单变量分析显示**真实、教科书级的疾病信号**（T2D→IDL 胆固醇酯、CHD→LDL 甘油三酯…）
- 一切"按疾病分配融合权重"的方案（手动规则/学习门控）全部失败：
  64 个 support 样本下互补性信号不可识别 —— few-shot 多组学的共性障碍

## 快速开始

```bash
pip install -r requirements.txt
export PROMETA_DATA_ROOT=/path/to/data   # 见「数据准备」

# Stage 1: 预训练 (相互预测 + 自重建), 两套机制各一条命令
python prometa201/pretrain_cls_dual.py --epochs 50
python prometa201/pretrain_cls_ca.py   --epochs 50

# Stage 2: MAML 疾病分类 (最优配置示例)
python prometa201/train_maml_cls_dual.py --gpu_id 0 --init cls \
    --cl_weight 0.1 --cl_temperature 0.07 --random_seed 42
python prometa201/train_maml_cls_ca.py   --gpu_id 0 --init none \
    --cl_weight 0.1 --cl_temperature 0.07 --random_seed 42

# 3 seeds 队列示例
bash scripts/run_3seeds_example.sh
```

每个训练任务约占 15GB 显存；多卡并行时一卡一任务。

## 数据准备

代码默认从 `PROMETA_DATA_ROOT`（未设置时为 `/data/zhangzimo/MedicalAI`）读取 ProMeta 原版
pipeline 的中间产物：

| 路径（相对 DATA_ROOT） | 内容 |
|---|---|
| `Prometa2.0_Plus/preprocessed_v2/` | `X_prot_{train,val}.npy` [N,2923]、`X_meta_{train,val}.npy` [N,251]、`eids_{train,val}.npy`、`meta_names.txt`（按 EID 对齐的双组学特征） |
| `prometa-meta/prometa_reproduce/data_correct/` | `term2pre_{cases,controls}_{train,valid,test}.pkl`（781 个疾病的病例/对照 EID 列表，train/valid/test 按病人切分） |
| `Prometa2.0_Plus/pathway_data/CPDB_pathways_genes.tab` | KEGG 通路-基因注释（蛋白 tokenizer 的通路掩码） |
| `Prometa2.0_clean_split/pathway_data/prot_names.txt` | 2923 个蛋白名（与特征列序一致） |

## 代码目录

```
prometa2.0.1/
├── README.md                      ← 本文件
├── requirements.txt
├── prometa201/                    ← 全部代码包
│   ├── paths.py                   ← 数据/输出路径 (PROMETA_DATA_ROOT 可覆盖)
│   ├── layers.py                  ← MetaTransformerLayer (transformer块), FocalLoss
│   ├── omics_encoders.py          ← PathwayGatedTokenizer / SimpleMetaEncoder / CrossAttentionBlock / load_pathway_mask
│   ├── dataset.py                 ← DualOmicsDataset (每病一个 few-shot 任务)
│   ├── maml.py                    ← MAML 框架: train_step / evaluate / InfoNCE / 指标 / 存结果
│   ├── models_cls_dual.py         ← 机制一: PretrainCLSDual + CLSDualMamlModel
│   ├── models_cls_ca.py           ← 机制二: CABlock + PretrainCLSDualCA + CLSDualCAMamlModel
│   ├── pretrain_cls_dual.py       ← Stage 1 入口 (CLSDual 预训练)
│   ├── pretrain_cls_ca.py         ← Stage 1 入口 (CLSCA 预训练)
│   ├── train_maml_cls_dual.py     ← Stage 2 入口 (CLSDual MAML, 含 --cl_weight/--mod_gate)
│   ├── train_maml_cls_ca.py       ← Stage 2 入口 (CLSCA MAML)
│   └── analysis/                  ← RQ2 机制分析脚本
│       ├── extract_rq2.py             (gate + 置换重要性)
│       ├── extract_rq2_saliency.py    (逐病输入梯度 saliency)
│       └── meta_univariate_auc.py     (代谢物单变量 AUROC)
├── docs/
│   ├── ARCHITECTURE.md            ← 小白向架构与代码导览 (每一层/每个 loss 在哪)
│   ├── RESULTS.md                 ← 完整实验结果与结论
│   └── RQ2_REPORT.md              ← RQ2 机制分析报告 (91 病逐病速查表)
├── results/
│   ├── results_per_disease_ALL_methods.{csv,md}   ← 逐病 AUROC 全方法对照表 (3-seed 逐病平均)
│   └── runs/                                    ← 全部 68 次运行的原始 JSON + MANIFEST.md 清单
└── scripts/
    └── run_3seeds_example.sh
```

新手建议阅读顺序：[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) → 跑通 Stage 1+2 → [docs/RESULTS.md](docs/RESULTS.md)。

**可复现性验证**：文档中每个数字都可由 `results/runs/` 下的原始 JSON 直接复算
（seeds、配置、逐病指标俱全）；逐病 CSV 与各 JSON 的一致性已程序化验证（误差=0）；
代码与原版实验代码逐函数 AST 比对一致（仅文档字符串差异）。
