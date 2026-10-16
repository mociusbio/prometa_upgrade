#!/bin/bash
# 示例: 3 seeds 跑 CLSCA+InfoNCE (本研究最优配置), 串行排队
# 实际实验中我们在 4 张 GPU 上并行 (每任务 ~15GB 显存, 一卡一任务);
# 若只有单卡, 直接串行执行即可。
set -e
cd "$(dirname "$0")/.."

for seed in 42 520 114514; do
  echo "=== seed $seed ==="
  python prometa201/train_maml_cls_ca.py \
    --gpu_id 0 --random_seed $seed \
    --init ca \
    --cl_weight 0.1 --cl_temperature 0.07
done
