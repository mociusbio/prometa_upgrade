# 实验原始结果清单（results/runs/）

每个 JSON 是一次完整训练运行的存档（`train_maml_*` / `MetaOnly` / `ProMeta` 脚本自动产出），
含 `config`（超参）、`summary_metrics`（宏平均）和 `per_task_details`（91 病逐病指标）。
`manifest.json` 是机读索引（含原始文件名，便于溯源）。

## 目录结构与对应论文数字

| 目录 | 实验 | seeds (Test AUROC) | mean | 对应文档 |
|---|---|---|---|---|
| `baseline_protein_only/` | 蛋白-only 基线 (原版 ProMeta) | 0.7052 / 0.7114 / 0.7040 / 0.7107 / 0.7040 (sup32, 5 seeds) | **0.7071** | README/RESULTS §1 |
| | + support 扫描 | sup16/8/4/2 = 0.7030/0.6804/0.6774/0.6626 | | RESULTS §5 |
| `meta_only/` | 代谢-only | 0.6225 / 0.6014 / 0.5863 (canonical 3 seeds) | 0.6034 | RQ2 报告 §3 |
| `clsdual_pre_v1_64d/` | CLSDual + v1 预训练 | 0.7054 / 0.7154 / 0.7102 | **0.7103** | RESULTS §1 |
| `clsdual_scratch_64d/` | CLSDual 无预训练 | 0.6980 / 0.7116 / 0.6985 | 0.7027 | RESULTS §1 |
| `clsdual_pre_v2_64d/` | CLSDual + v2_64d 预训练 | 0.7051 / 0.7166 / 0.7077 | 0.7099 | RESULTS §2 配对基线 |
| `clsdual_d128/` | CLSDual d_model=128 | 0.7054 / 0.7135 / 0.7114 | 0.7101 | RESULTS §6 |
| `clsdual_infonce_l0.1_t0.07/` | +InfoNCE λ=0.1 τ=0.07（最优） | 0.7108 / 0.7141 / 0.7138 | **0.7129** | RESULTS §2 |
| `clsdual_infonce_l1.0_t0.07/` | +InfoNCE λ=1.0 | 0.6969 / 0.7085 / 0.7160 | 0.7071 | RESULTS §2 |
| `clsdual_infonce_l0.1_t0.1/` | +InfoNCE τ=0.1 | 0.7126 / 0.7125 / 0.7108 | 0.7120 | RESULTS §2 |
| `clsdual_infonce_l0.1_t0.5/` | +InfoNCE τ=0.5 | 0.7098 / 0.7165 / 0.7063 | 0.7109 | RESULTS §2 |
| `clsdual_modgate/` | +mod_gate（阴性） | 0.7057 / 0.7116 / 0.7065 | 0.7079 | RESULTS §2 |
| `clsdual_support_scan/` | CLSDual support 扫描 | sup16/8/4/2 = 0.7018/0.6890/0.6770/0.6769 | | RESULTS §5 |
| `clsca_pre_v1_64d/` | CLSCA + 预训练 | 0.7021 / 0.7085 / 0.7012 | 0.7039 | RESULTS §1 |
| `clsca_scratch_64d/` | CLSCA 无预训练（全局最优） | 0.7123 / 0.7078 / 0.7183 | **0.7128** | RESULTS §1 |
| `clsca_pre_v2_64d/` | CLSCA + v2_64d 预训练 | 0.7098 / 0.7000 / 0.7045 | 0.7048 | RESULTS §6 |
| `clsca_d128/` | CLSCA d_model=128 | 0.6921 / 0.7075 / 0.6968 | 0.6988 | RESULTS §6 |
| `clsca_modgate/` | +mod_gate（阴性） | 0.7082 / 0.7053 / 0.7116 | 0.7084 | RESULTS §2 |
| `clsca_infonce_modgate/` | +InfoNCE+mod_gate（叠加有害） | 0.6963 / 0.7090 / 0.6946 | 0.7000 | RESULTS §2 |
| `clsca_support_scan/` | CLSCA support 扫描 | sup16/8/4/2 = 0.7095/0.6986/0.6836/0.6657 | | RESULTS §5 |

## 备注

- **seeds**：42 / 520 / 114514（1919810、20260121 仅用于蛋白/代谢基线的 5-seed 版本）。
- **预训练 ckpt 版本**：v1 = `best_model.pt`（0911 批次）；v2_64d = `checkpoints_pretrain/best_model_v2_64d.pt`
  （0917 批次重训）。JSON 的 `extra.init_ckpt` 字段记录了每次运行实际使用的 ckpt 路径。
- **作废运行**：`meta_only/sup32_seed42_superseded_20260910_030341.json`（宏平均仅 0.4582，被 041142 替代，
  保留仅供溯源）；ProMeta seed20260121 有两次 0.5211 的坍缩运行未收录（修复后 = 0.7040）。
- **0917 批次的划分**：时间戳 `03xxxx` = d64 v2；`07/08/09xxxx` = d128（见 RESULTS §6）。
- **InfoNCE/mod_gate 配置不在 JSON config 中**（旧版脚本未记录），以本清单的目录名为准。
- `results/results_per_disease_ALL_methods.csv` 的每列 = 对应实验 3-seed 的**逐病平均**，
  已程序化验证与该目录下的 JSON 完全一致（平均绝对差 = 0）。
- **mod_gate 塌缩的完整证据**（每疾病学到的融合权重 = JSON 里的 `mod_gate_w` 字段）：
  见 [MODGATE_COLLAPSE.md](MODGATE_COLLAPSE.md)——6 个运行各自塌缩为常数（w_meta 0.499~0.565），
  91 病跨病 std ≈ 0.0001，从未逐病分化。
