# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

多目标贝叶斯网络结构学习。使用 MOEA/D + 切比雪夫分解同时优化两个目标：MDL 评分（描述长度最小化）和与先验网络的结构对称差。所有图结构保证无环（DAG）。

## 常用命令

```bash
# 环境配置
bash scripts/setup_venv.sh && source venv/bin/activate

# 单次运行
python3 main.py --model asia
python3 main.py --model alarm --pop-size 100 --generations 500 --max-sdiff 50 --n-samples 5000 --max-parents 4

# Batch 并行 (同一先验+数据，N 次运行汇总)
python3 main.py --model alarm --batch 30 --workers 8 \
    --pop-size 100 --generations 10000 --n-samples 5000 \
    --max-sdiff 50 --max-parents 4 --output ./output/batch_alarm

# Profiling 交叉算子性能
python3 -m cProfile -s cumulative profile_crossover.py 2>&1 | head -80
```

## 架构: 自底向上的模块依赖链

```
config.py ──→ graph.py ──→ score.py
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
- `_would_create_cycle(u,v)`: **无界 DFS** 检查添加边是否产生任意环（禁止所有环，保证 DAG）
- `_iter_cycles()`: DFS 枚举所有简单环，`min(path)==start` 去重
- 无 `max_forbidden_cycle` 参数——已删除，统一禁止所有环

### 评分 (`score.py`)
- **MDL = -BIC**（越小越好）。关键：曾出现符号反转 bug（BIC 越大越好但 MOEA/D 最小化目标）
- **可分解评分**: 每个节点评分仅依赖其父集
- **无 CompositeScore**——已删除（dirty-node 机制从未被使用）
- **全局组合缓存**: `MOEAD._score_cache: dict[(node, frozenset(parents)), (mdl, sdiff)]`，一次 lookup 拿两个值。交叉算子和 `_evaluate` 均走此缓存

### 切比雪夫聚合 (`decomposition.py`)
- 2 目标 Das-Dennis: H 分区 → H+1 个权重向量
- 归一化: `g = max_i { λ_i * |f_i - z*_i| / max(nadir_i - ideal_i, ε) }`
- ideal 和 nadir 每代从种群动态更新

### 遗传算子 (`operators.py`)
- **交叉**: **子问题感知交叉**。逐节点用归一化切比雪夫聚合（嵌入当前子问题的 λ）比较两父代父集，选更优者组装。从空图构建，即时判环。这是核心创新——同一对父代，不同子问题产生不同子代
- **交叉模式** (`crossover_type`): `sequential`（默认，节点按自然索引序，利用 bnlearn 网络的隐式拓扑序让判环 DFS 瞬间触底）| `no-cycle-check`（不判环直接构建，事后 `_fix_cycles` 随机删边修复——仅保留供论文对比，大图上不可行）
- **变异**: 随机 k ∈ [mutation_ops_min, mutation_ops_max]（默认 2-6）次操作，等概率加边/删边/反转边。动态范围在同一世代内混合微调与跳跃两种策略
- **父代选择**: 概率 δ 从邻居选，否则全局选

### MOEA/D 主循环 (`moead.py`)
1. 生成权重向量 → 计算邻居 B[i]
2. 初始种群（先验扰动）
3. 评估 F[i] = (mdl, sdiff)（走组合缓存）
4. 每代对每个子问题: 选父代 → 交叉 → 变异 → 评估 → 更新 ideal → 更新邻居
5. 提取非支配解 → MOEADResult

### 先验网络 (`prior.py`)
- bnlearn 模型: 加载原始图 → 随机变异 6 次作为先验（避免标准答案）
- BIF 文件: 直接使用，不做变异
- `generate_data()` 从 pgmpy 模型采样合成数据

## 重要实验结论

- **MDL 自然约束 Sdiff**: max_sdiff 从 20 变到 500，实际 Sdiff 始终 ≤16。设 50 或自动计算均可
- **禁止所有环最优**: 无界 DFS 100% 无环且最快（比限界 BFS 快 2.7x）
- **全局缓存 8.78x 加速**: 组合缓存 + 删除 CompositeScore 后，batch_alarm 从 17,100s 降到 1,948s
- **变异 2-6 最优**: 8-12 导致种群散乱，退化为随机搜索
- **节点自然序 = 隐式拓扑序**: bnlearn 网络节点索引遵循因果序（原因→结果），按此序 `add_edge` 时子节点尚无后代，DFS 瞬间触底。随机打乱破坏此序 → 3.29x 减速（andes 223 节点）。`sequential` 模式利用此性质，是所有规模上的最优选择
- **边阻塞率 0.06-0.65%，但极其关键**: 每条被拒边都是"闭环边"，阻止它防止了整个子代变成含环图（阻止率 18-25%）。判环开销微不足道（每次 DFS 仅遍历 ~7 节点），远小于事后修复环的代价
- **no-cycle-check 大图上不可行**: `_fix_cycles` 复杂度 O(edges² × n)，andes (223 节点) 上 >12h 未完成。仅保留供论文对比实验
- 详见 `experiment-conclusions.md`

## 重要注意

- 本项目使用 **pickle** 序列化结果对象 (MOEADResult)
- `--no-plot` 跳过单次运行的可视化；batch 汇总图始终生成
- 网络结构图需 `--plot-networks` 才输出（默认关闭）
- Batch 模式 `--no-params` 内部使用，跳转子进程 params.json 生成
- `matplotlib.use("Agg")` 在 visualize.py 顶部设置，确保无头环境可用
- Graphviz dot 用于网络结构图渲染（`graphviz` Python 包）
- `params.json` 自动包含 `git_commit` 字段，记录实验时的代码版本
- 无测试套件；验证靠单次运行和 batch 实验
