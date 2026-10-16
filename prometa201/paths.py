# -*- coding: utf-8 -*-
"""统一数据路径配置。

所有脚本的数据路径都从这里读。默认指向本机原始数据根目录；
在其他机器上运行时设置环境变量 PROMETA_DATA_ROOT 即可, 无需改代码:

    export PROMETA_DATA_ROOT=/path/to/your/data

所需数据(均为 ProMeta 原版流程的产物, 见 README「数据准备」):
  - preprocessed_v2/       按 EID 对齐的蛋白/代谢特征矩阵 (.npy)
  - data_correct/          train/valid/test 三 split 的病例/对照 EID 列表 (.pkl)
  - CPDB_pathways_genes.tab  KEGG 通路-基因注释 (蛋白 tokenizer 的通路掩码)
  - prot_names.txt         蛋白名列表 (与特征矩阵列顺序一致)
"""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get('PROMETA_DATA_ROOT', '/data/zhangzimo/MedicalAI')

PREPROCESSED_V2_DIR = os.path.join(DATA_ROOT, 'Prometa2.0_Plus', 'preprocessed_v2')
PKL_INPUT_DIR = os.path.join(DATA_ROOT, 'prometa-meta', 'prometa_reproduce', 'data_correct')
CPDB_FILE_PATH = os.path.join(DATA_ROOT, 'Prometa2.0_Plus', 'pathway_data', 'CPDB_pathways_genes.tab')
PROT_NAMES_FILE = os.path.join(DATA_ROOT, 'Prometa2.0_clean_split', 'pathway_data', 'prot_names.txt')

# 所有运行输出(checkpoints / logs / results)统一放仓库内 outputs/, 已被 .gitignore 忽略
OUTPUT_DIR = os.path.join(REPO_ROOT, 'outputs')
