"""ProMeta 2.0.1 — 双组学（蛋白+代谢）few-shot 疾病预测的机制研究与基线代码。

两套融合机制:
  - CLSDual: 双分支 transformer, CLS 拼接 (models_cls_dual.py)
  - CLSCA:   双分支 Cross-Attention, token 池化拼接 (models_cls_ca.py)
附加组件: InfoNCE 对比学习 / 可学习模态门控 (maml.py train_step 内, 默认关闭)

数据流: Stage 1 预训练(相互预测+重建) -> Stage 2 MAML 疾病分类
"""
__version__ = "2.0.1"
