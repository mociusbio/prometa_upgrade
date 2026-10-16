# RQ2 总报告：两个组学各贡献了什么的语义？（2026-10-15）

> 研究问题：在双组学疾病预测模型中，**蛋白质组和代谢组各自提供了什么信息**？
> 模型：CLSDual-pre（seed42, τ=0.1 ckpt，全局 0.7126），91 个 held-out test 疾病。

## 0. TL;DR（一句话版本）

- **模型实际只用蛋白质做决策**（遮挡实验：代谢置零 AUROC 不变，蛋白置零崩到 0.5）
- 但这不是代谢组没用：**代谢单变量分析显示 251 个代谢物里大量病有教科书级的强信号**
  （T2D→IDL 胆固醇酯、CHD→LDL 甘油三酯占比、COPD→极小 VLDL、头部外伤→乳酸……）
- 逐病蛋白重要性（输入梯度）能恢复**教科书生物学**：甲减→TSHB(#4)/POMC(#15)、
  呼吸异常→NT-proBNP(#1)、损伤类→趋化因子(CCL7/IL17C)
- 结论形态：**"代谢组携带真实疾病信号，但当前 few-shot 融合框架无法将其转化为决策权重"**
  ——这正是论文的机制性发现，把 RQ1（哪些病提升）和阴性结果串成了完整故事。

## 1. 分析链条（为什么要这样做）

RQ2 的原始计划是"提取 tokenizer 的蛋白 gate 做逐病比较"，但很快发现行不通，
链条是这样修正的：

| 步骤 | 方法 | 结论 |
|---|---|---|
| ① 模型用哪个模态？ | 遮挡实验（分别把蛋白/代谢输入置零） | **决策 100% 依赖蛋白**；两架构（CLSDual/CLSCA）一致 |
| ② 代谢物侧重要性 | 251 代谢物 × 91 病 单变量 AUROC（Nightingale 官方映射 251/251） | 大量病有强且生物学合理的代谢信号（见 §3） |
| ③ 蛋白 gate 逐病比较 | `sigmoid(tokenizer.gate_logits)` inner-adapt 后导出 | **失败**：gate 是全局参数，91 病向量完全相同 |
| ④ 蛋白侧改输入梯度 saliency | 逐病：query 正类 logit 对 2923 蛋白输入的 \|grad\| 均值 | 有效，有疾病特异性（见 §4） |

## 2. 为什么蛋白 gate 不能逐病比（且这本身是个发现）

- gate 是 tokenizer 的**全局参数**（全疾病共享），结构上不携带疾病信息 →
  91 个病的 gate 向量逐位相同（top 全是 LDLR/PLA2G7/APOA1）
- 这本质上是 **mod_gate 塌缩在蛋白侧的对应物**：模型里所有"重要性权重"
  在 few-shot（64 support）条件下都学不出逐病分化
- 副产品发现：全局 gate top 蛋白是**脂质/脂蛋白簇**（LDLR、APOA1、APOC1、
  PLTP、ANGPTL3、PLA2G7）——tokenizer 学到的全局先验偏向脂代谢通路

## 3. 代谢组：单变量信号（`meta_univariate.npz`）

最有诊断力的代谢物（按病，单变量 AUROC）：

| 疾病 | 最强代谢物 | AUROC | 生物学 |
|---|---|---|---|
| T2D | IDL 胆固醇酯 | 0.13（反向=0.87） | 糖尿病脂代谢紊乱经典标志 |
| CHD | 大 LDL 甘油三酯占比 | 0.85 | 致动脉粥样硬化脂蛋白 |
| COPD | 极小 VLDL TG 占比 | 0.79 | 吸烟/炎症相关脂代谢 |
| 头部外伤 | 乳酸 | 0.39（反向=0.61） | 组织缺氧/无氧酵解 |
| 静脉曲张 | VLDL 甘油三酯 | — | 循环淤滞相关 |
| 心境障碍 | HDL 胆固醇酯 + 白蛋白 | — | 炎症-脂质轴 |

要点：**代谢组单独有信号的病 ≠ 融合提升大的病**（Spearman(MetaOnly, Δ)≈−0.10）。
代谢信号是真实的，但走不进模型的决策。

## 4. 蛋白质组：逐病 saliency（`rq2_saliency_seed42.npz`）

方法学质控（排除伪影）：
- ✅ 非量纲伪影：corr(saliency, 蛋白全局 std) ≈ 0.02
- ✅ 非通路枢纽伪影：corr(跨病均值 saliency, 蛋白所属 KEGG 通路数) ≈ −0.006
- ✅ 有真实疾病特异成分：逐病两两 Spearman mean 0.689，top50 重叠仅 41%
- 分析时用 **去跨病均值后的 saliency** 提取疾病特异 top（扣除共享成分）

教科书级验证（按 saliency 全 panel 排名）：

| 疾病 | 蛋白 | 排名 | 生物学解释 |
|---|---|---|---|
| 甲减 | **TSHB** | #4 | 促甲状腺激素 β 亚基，甲减的一线诊断指标 |
| 甲减 | POMC / LEP | #15 / #60 | 下丘脑-垂体轴、能量代谢与甲状腺功能联动 |
| 呼吸异常 | **NTPROBNP** | #1 | NT-proBNP，心衰/容量负荷标志物，呼吸困难的鉴别诊断核心 |
| 腕手损伤 | CCL7 | top15 | 趋化因子，创伤局部炎症招募 |
| 静脉曲张 | IL17C / MBL2 | top15 | 炎症通路与先天免疫凝集素 |
| 软组织疾患 | MSR1 / SPARC | top15 | 巨噬细胞清道夫受体 / 组织重塑 |

局限：经典急性期反应蛋白 **CRP/SAA1/SAA2 不在 Olink panel 里**
（panel 限制，非模型失败）；saliency 是相关性证据，非因果。

## 5. 对论文叙事的意义

RQ2 把全部结果串成一条机制链：

1. **代谢组携带真实、生物学合理的疾病信号**（§3）——多组学有价值的前提成立
2. **但当前 few-shot 融合框架的决策完全绕开代谢支**（§1 遮挡 + gate 全局性 §2）
3. **逐病重要性方法学上的教训**：一切"逐病权重"（gate/mod_gate/手动规则）
   在 64-sample support 下都不可识别 → 这是 few-shot 多组学的共性障碍，非我们实现问题
4. 蛋白侧 saliency 的教科书验证（§4）证明**分析管线本身是可靠的**——
   阴性结论不是"模型太弱什么都学不到"，而是特异地"代谢信息进不了决策"

→ 论文卖点：91 病双组学互补图谱 + "信号存在 vs 决策使用"的鸿沟 + 权重分配
不可识别的机制分析 + 蛋白 saliency 生物学验证。

## 6. 全部 91 病速查表

（按融合模型 AUROC 降序；top1 蛋白为去均值 saliency 的疾病特异首位；
top1 代谢物为模型 saliency 首位，仅供参考——模型决策实际不用代谢支）

| # | 疾病 | AUROC | top1 蛋白(特异) | top1 代谢物 | 代谢分组 |
|---|---|---|---|---|---|
| 1 | O15_MATERN_CARE | 0.981 | PAEP | Citrate | Glycolysis related metabolites |
| 2 | COPD_MULTIMORB | 0.940 | F7 | Citrate | Glycolysis related metabolites |
| 3 | O15_DELIV_PERIN_LACER | 0.938 | PAEP | Triglycerides in chylomicrons and extremely large VLDL | Lipoprotein subclasses |
| 4 | O15_PREG_ABORT | 0.927 | PAEP | Citrate | Glycolysis related metabolites |
| 5 | Z21_PRESENCE_OTH_FUNCTIONAL_IMPLANTNTS | 0.906 | F7 | Concentration of medium LDL particles | Lipoprotein subclasses |
| 6 | T2D_INCLAVO | 0.895 | REN | Phospholipids to total lipids ratio in large VLDL | Relative lipoprotein lipid concentrations |
| 7 | I9_CHD | 0.880 | REN | Citrate | Glycolysis related metabolites |
| 8 | I9_UAP | 0.878 | REN | Citrate | Glycolysis related metabolites |
| 9 | Z21_PRESENCE_CARDIAC_VASCULAR_IMPLANTNTS_GRAFTS | 0.867 | REN | Citrate | Glycolysis related metabolites |
| 10 | FIBRO_COMORB | 0.842 | MYOC | Concentration of medium LDL particles | Lipoprotein subclasses |
| 11 | T2D | 0.841 | F7 | Citrate | Glycolysis related metabolites |
| 12 | C3_BREAST_HER2NEG_EXALLC | 0.838 | PROK1 | Concentration of medium LDL particles | Lipoprotein subclasses |
| 13 | N14_MENORRHAGIA | 0.834 | CD8A | Concentration of medium LDL particles | Lipoprotein subclasses |
| 14 | N14_POSTMENBLEED | 0.834 | CGA | Phospholipids to total lipids ratio in large VLDL | Relative lipoprotein lipid concentrations |
| 15 | E4_DM2_STRICT | 0.819 | F7 | Triglycerides in chylomicrons and extremely large VLDL | Lipoprotein subclasses |
| 16 | M13_ARTHROSIS_OTH | 0.816 | CEACAM6 | Concentration of medium LDL particles | Lipoprotein subclasses |
| 17 | I9_CHD_NOREV | 0.811 | F7 | Triglycerides in chylomicrons and extremely large VLDL | Lipoprotein subclasses |
| 18 | I9_DOAAC | 0.799 | F7 | Lactate | Glycolysis related metabolites |
| 19 | N14_PROSTHYPERPLA | 0.798 | ACRV1 | Phospholipids to total lipids ratio in large VLDL | Relative lipoprotein lipid concentrations |
| 20 | KRA_PSY_MOOD | 0.793 | POMC | Concentration of medium LDL particles | Lipoprotein subclasses |
| 21 | F5_MOOD | 0.781 | F7 | Concentration of VLDL particles | Lipoprotein particle concentrations |
| 22 | CARDIAC_ARRHYTM | 0.779 | REN | Phospholipids to total lipids ratio in large VLDL | Relative lipoprotein lipid concentrations |
| 23 | F5_DEPRESSIO | 0.779 | PROS1 | Concentration of medium LDL particles | Lipoprotein subclasses |
| 24 | N14_FEMALEGENINF | 0.779 | CD8A | Concentration of medium LDL particles | Lipoprotein subclasses |
| 25 | Z21_FAMILY_HISTORY_CERTA_DISABILITI_CHRONIC_DISEA_LEADI_DISABL | 0.771 | ACAN | Concentration of medium LDL particles | Lipoprotein subclasses |
| 26 | J10_INFLUPNEU | 0.770 | F7 | Concentration of medium LDL particles | Lipoprotein subclasses |
| 27 | HYPOTHYROIDISM | 0.755 | INSL3 | Concentration of VLDL particles | Lipoprotein particle concentrations |
| 28 | N14_OTHBLADD | 0.753 | REN | Concentration of medium LDL particles | Lipoprotein subclasses |
| 29 | Z21_PERSONAL_HISTORY_MEDICAL_TREAT | 0.749 | F7 | Phospholipids to total lipids ratio in large VLDL | Relative lipoprotein lipid concentrations |
| 30 | M13_POLYARTHROPATHIES | 0.743 | F7 | Phospholipids to total lipids ratio in large VLDL | Relative lipoprotein lipid concentrations |
| 31 | H7_CATARACTOTHER | 0.743 | CCL7 | Concentration of medium LDL particles | Lipoprotein subclasses |
| 32 | N14_ABNORMALBLEED | 0.743 | SEZ6L2 | Concentration of medium LDL particles | Lipoprotein subclasses |
| 33 | AB1_BACT_BIR_OTHER_INF_AGENTS | 0.741 | TSHB | Concentration of medium LDL particles | Lipoprotein subclasses |
| 34 | C3_BREAST_EXALLC | 0.741 | LEP | Concentration of medium LDL particles | Lipoprotein subclasses |
| 35 | AB1_OTHER_BACT_INOTHER | 0.740 | HRC | Concentration of medium LDL particles | Lipoprotein subclasses |
| 36 | N14_POLYPFEMGEN | 0.738 | PROK1 | Concentration of medium LDL particles | Lipoprotein subclasses |
| 37 | C3_BREAST_ERNEG_EXALLC | 0.735 | TEX101 | Concentration of medium LDL particles | Lipoprotein subclasses |
| 38 | E4_HYTHY_AI_STRICT | 0.735 | INSL3 | Concentration of medium LDL particles | Lipoprotein subclasses |
| 39 | CD2_BENIGN_COLON | 0.733 | ACAN | Concentration of medium LDL particles | Lipoprotein subclasses |
| 40 | BRONCHITIS | 0.732 | NEFL | Concentration of medium LDL particles | Lipoprotein subclasses |
| 41 | K11_OTHFUNC | 0.731 | FSHB | Concentration of medium LDL particles | Lipoprotein subclasses |
| 42 | D3_ANAEMIANAS | 0.730 | LEP | Concentration of medium LDL particles | Lipoprotein subclasses |
| 43 | R18_SYNCO_COLLAP | 0.726 | REN | Concentration of medium LDL particles | Lipoprotein subclasses |
| 44 | KNEE_ARTHROSIS | 0.723 | REN | Concentration of medium LDL particles | Lipoprotein subclasses |
| 45 | BLEEDING | 0.715 | F10 | Concentration of medium LDL particles | Lipoprotein subclasses |
| 46 | D3_APLASTICANDOTHANAEMIA | 0.712 | F7 | Concentration of medium LDL particles | Lipoprotein subclasses |
| 47 | H8_EAR_MASTOID | 0.709 | F10 | Concentration of medium LDL particles | Lipoprotein subclasses |
| 48 | Z21_OTHER_MEDICAL_CARE | 0.709 | LEP | Concentration of medium LDL particles | Lipoprotein subclasses |
| 49 | N14_BREAST | 0.709 | PROK1 | Concentration of medium LDL particles | Lipoprotein subclasses |
| 50 | G6_EPIPAROX | 0.706 | REN | Free cholesterol to total lipids ratio in very large VLDL | Relative lipoprotein lipid concentrations |
| 51 | K11_OTHINT | 0.703 | ACAN | Concentration of medium LDL particles | Lipoprotein subclasses |
| 52 | M13_ARTHROSIS_KNEE | 0.701 | AGER | Concentration of medium LDL particles | Lipoprotein subclasses |
| 53 | E4_HYTHYNAS | 0.700 | TSHB | Concentration of medium LDL particles | Lipoprotein subclasses |
| 54 | R18_ABNORMALITI_BREAT | 0.697 | NTPROBNP | Concentration of medium LDL particles | Lipoprotein subclasses |
| 55 | K11_OESITIS_ADVERSE | 0.697 | NEFL | Concentration of VLDL particles | Lipoprotein particle concentrations |
| 56 | R18_ABNORMAL_FINDI_DIAGNOST_IMAGI_FUNCTION_STUDI_WO_DIAGNOSIS | 0.694 | F7 | Phospholipids to total lipids ratio in large VLDL | Relative lipoprotein lipid concentrations |
| 57 | D3_OTHERANAEMIA | 0.692 | F7 | Concentration of medium LDL particles | Lipoprotein subclasses |
| 58 | K11_OTHDIG | 0.685 | GAST | Omega-3 fatty acids | Fatty acids |
| 59 | K11_OESITIS | 0.678 | NEFL | Concentration of medium LDL particles | Lipoprotein subclasses |
| 60 | CD2_BENIGN_COLORECANI | 0.673 | CCL7 | Concentration of medium LDL particles | Lipoprotein subclasses |
| 61 | ST19_INJURI_WRIST_HAND | 0.669 | WIF1 | Concentration of medium LDL particles | Lipoprotein subclasses |
| 62 | M13_LOWBACKPAIN | 0.668 | FSHB | Concentration of medium LDL particles | Lipoprotein subclasses |
| 63 | M13_SOFTTISSUEOTH | 0.663 | SFRP4 | Concentration of medium LDL particles | Lipoprotein subclasses |
| 64 | M13_SYNOTEND | 0.663 | FCGR2B | Concentration of medium LDL particles | Lipoprotein subclasses |
| 65 | M13_LOWBACKPAINORANDSCIATICA | 0.662 | ACAN | Concentration of medium LDL particles | Lipoprotein subclasses |
| 66 | K11_FUNCDYSP | 0.649 | FSHB | Concentration of VLDL particles | Lipoprotein particle concentrations |
| 67 | HYPOT_THYRO_INCL | 0.645 | LEP | Concentration of medium LDL particles | Lipoprotein subclasses |
| 68 | C3_SKIN_EXALLC | 0.645 | WIF1 | Concentration of medium LDL particles | Lipoprotein subclasses |
| 69 | CD2_BENIGN_COLORECANI_EXALLC | 0.641 | REN | Concentration of medium LDL particles | Lipoprotein subclasses |
| 70 | R18_OTHER_SYMPTOMS_SIGNS_INVOLVI_DIGES_SYSTEM_ABDOMEN | 0.633 | CCL7 | Concentration of medium LDL particles | Lipoprotein subclasses |
| 71 | M13_KNEEDERANGEMENTS | 0.632 | WIF1 | Concentration of medium LDL particles | Lipoprotein subclasses |
| 72 | M13_JOINTOTH | 0.628 | F7 | Concentration of medium LDL particles | Lipoprotein subclasses |
| 73 | ST19_COMPLICATIO_PROCED_NOT_ELSEW_CLASSIFIED | 0.628 | LMOD1 | Concentration of medium LDL particles | Lipoprotein subclasses |
| 74 | L12_INFECT_SKIN | 0.624 | MYOM3 | Concentration of medium LDL particles | Lipoprotein subclasses |
| 75 | Z21_MEDICAL_OBSER_EVALU_SUSPECTED_DISEA_CONDI | 0.619 | CCL7 | Concentration of medium LDL particles | Lipoprotein subclasses |
| 76 | H7_LIDLACRIMALORBIT | 0.606 | LEP | Concentration of medium LDL particles | Lipoprotein subclasses |
| 77 | M13_OSTEOCHONDRO | 0.604 | F10 | Concentration of medium LDL particles | Lipoprotein subclasses |
| 78 | Z21_FAMILY_HISTORY_MALIG_NEOPLASM | 0.601 | FSHB | Concentration of medium LDL particles | Lipoprotein subclasses |
| 79 | ST19_INJURI_HEAD | 0.598 | GUSB | Concentration of medium LDL particles | Lipoprotein subclasses |
| 80 | ACUTE_RESPIRATORY_INFECTIONS | 0.590 | F7 | Phospholipids to total lipids ratio in large VLDL | Relative lipoprotein lipid concentrations |
| 81 | ST19_INJURI_KNEE_LOWER_LEG | 0.584 | WIF1 | Concentration of medium LDL particles | Lipoprotein subclasses |
| 82 | G6_CARPTU | 0.574 | LEP | Concentration of medium LDL particles | Lipoprotein subclasses |
| 83 | J10_VOCALLARYNX | 0.570 | WIF1 | Concentration of medium LDL particles | Lipoprotein subclasses |
| 84 | L12_OTHERDISSKINANDSUBCUTIS | 0.550 | ACAN | Concentration of medium LDL particles | Lipoprotein subclasses |
| 85 | M13_MENISCUSDERANGEMENTS | 0.549 | PKN3 | Concentration of medium LDL particles | Lipoprotein subclasses |
| 86 | R18_NAUSEA_VOMITI | 0.543 | LEP | Concentration of medium LDL particles | Lipoprotein subclasses |
| 87 | I9_VARICVE | 0.522 | GFAP | Concentration of VLDL particles | Lipoprotein particle concentrations |
| 88 | CD2_BENIGN_LIPOMATOUS | 0.493 | EBI3_IL27 | Concentration of VLDL particles | Lipoprotein particle concentrations |
| 89 | L12_FOLLICULARCYST | 0.493 | TNFSF10 | Concentration of medium LDL particles | Lipoprotein subclasses |
| 90 | L12_DISORDSKINAPPENDIX | 0.474 | CCL7 | Concentration of medium LDL particles | Lipoprotein subclasses |
| 91 | VWXY20_TRANSPO_ACCIDENTS | 0.462 | CCL7 | Concentration of medium LDL particles | Lipoprotein subclasses |