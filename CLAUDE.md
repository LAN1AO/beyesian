# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

多目标贝叶斯网络结构学习。使用 MOEA/D + 切比雪夫分解同时优化两个目标：MDL 评分（描述长度最小化）和与先验网络的结构对称差。采用"放松约束搜索 + 后续修复"范式：搜索阶段允许长环（禁止 ≤max_forbidden_cycle 的短环），最终通过加权 FAS 算法转化为 DAG。

## 常用命令

```bash
# 环境配置
bash scripts/setup_venv.sh && source venv/bin/activate

# 单次运行
python3 main.py --model asia
python3 main.py --model alarm --pop-size 100 --generations 500 --max-sdiff 50 --n-samples 5000 --max-parents 4

# Batch 并行 (同一先验+数据，N 次运行汇总)
python3 main.py --model alarm --batch 20 --pop-size 100 --generations 10000 --n-samples 5000 --max-sdiff 50 --max-parents 4
```

## 架构: 自底向上的模块依赖链

```
config.py ──→ graph.py ──→ score.py ──→ score_cache.py
                │
                ├──→ prior.py (先验网络加载/扰动/数据生成)
                ├──→ decomposition.py (Das-Dennis 权重, 切比雪夫聚合, 非支配排序)
                ├──→ operators.py (交叉/变异/选择)
                └──→ population.py (初始种群生成)

moead.py ──→ 组装上述全部模块，实现 MOEA/D 主循环
visualize.py ──→ Pareto 前沿/收敛/网络结构可视化
main.py ──→ CLI 入口，组装 prior + moead + visualize
```

## 核心设计要点

### DirectedGraph (`graph.py`)
- `numpy` 邻接矩阵 `(n×n, int8)` 存储图结构
- `add_edge(u,v)` / `remove_edge(u,v)` / `reverse_edge(u,v)` 均带环检测
- `_would_create_cycle(u,v)`: 限界 BFS 检查添加边是否产生 ≤max_forbidden_cycle 的短环
- `_iter_cycles()`: DFS 枚举所有简单环，`min(path)==start` 去重
- `max_forbidden_cycle` 默认值 = `floor(sqrt(n_nodes))`，用户可自定义

### 评分 (`score.py` + `score_cache.py`)
- **MDL = -BIC**（越小越好）。BIC = 似然项 - 惩罚项。关键：曾出现过符号反转 bug（BIC 越大越好但 MOEA/D 最小化目标）
- **可分解评分**: 每个节点评分仅依赖其父集。修改边 (p→i) 时仅重算节点 i
- **CompositeScore**: 维护 `_mdl_cache[node]` / `_sdiff_cache[node]` + `_dirty_nodes` 集合。`scores_vector()` 仅重算 dirty 节点
- 交叉算子利用缓存: 传 `parent*_scores` 免重复计算

### 切比雪夫聚合 (`decomposition.py`)
- 2 目标 Das-Dennis: H 分区 → H+1 个权重向量
- 归一化: `g = max_i { λ_i * |f_i - z*_i| / max(nadir_i - ideal_i, ε) }`
- ideal 和 nadir 每代从种群动态更新

### 遗传算子 (`operators.py`)
- **交叉**: 逐节点比较两父代的父集评分（切比雪夫聚合），选更优父集组装子代；加入边时若产生非法环则跳过
- **变异**: 随机 2-6 次操作（等概率加边/删边/反转边）
- **父代选择**: 概率 δ 从邻居选，否则全局选

### MOEA/D 主循环 (`moead.py`, 核心约 150 行)
1. 生成权重向量 → 计算邻居 B[i]
2. 初始种群（先验扰动）
3. 评估 F[i] = (mdl, sdiff)
4. 每代对每个子问题: 选父代 → 交叉 → 变异 → 评估 → 更新 ideal → 更新邻居
5. 提取非支配解 → MOEADResult

### 先验网络 (`prior.py`)
- bnlearn 模型: 加载原始图 → 随机变异 6 次作为先验（避免标准答案）
- BIF 文件: 直接使用，不做变异
- `generate_data()` 从 pgmpy 模型采样合成数据

## 重要注意

- 本项目使用 **pickle** 序列化结果对象 (MOEADResult)。这是设计选择，对象图包含 numpy 数组和自定义类。
- `--no-plot` 仅跳过单次运行的可视化。Batch 模式的汇总图始终生成。
- `matplotlib.use("Agg")` 在 visualize.py 顶部设置，确保无头环境可用。
- Graphviz dot 用于网络结构图渲染（`graphviz` Python 包，非 `pygraphviz`）。
