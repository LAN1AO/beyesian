# 多目标贝叶斯网络结构学习 — 使用指南

## 环境准备

```bash
cd beyesian
bash scripts/setup_venv.sh   # 一键创建虚拟环境
source venv/bin/activate
```

## 基础用法

```bash
# 单次运行 — Asia 网络 (8节点)
python3 main.py --model asia

# 单次运行 — Alarm 网络 (37节点)
python3 main.py --model alarm --pop-size 100 --generations 500 \
    --max-sdiff 50 --n-samples 5000 --max-parents 4

# Batch 并行 — 同一先验+同一数据，跑 20 次汇总
python3 main.py --model alarm --batch 20 \
    --pop-size 100 --generations 10000 --n-samples 5000 \
    --max-sdiff 50 --max-parents 4
```

## 参数说明

### 先验网络
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `asia` | bnlearn 网络名，可选 `alarm`, `sachs`, `child`, `insurance` 等 |
| `--bif` | 无 | BIF 文件路径，优先级高于 `--model` |
| `--prior-file` | 无 | 预生成先验 (.pkl)，batch 模式自动使用 |

### 数据
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--n-samples` | `500` | 合成数据的样本数 |
| `--data-file` | 无 | 预生成数据 (.npy)，batch 模式自动使用 |

### MOEA/D 核心
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--pop-size` | `50` | 种群大小 / 权重向量数 |
| `--neighbors` | `10` | 邻居数量 T |
| `--generations` | `200` | 最大世代数 |
| `--prob-neighbor` | `0.9` | 从邻居选父代的概率 |
| `--max-replace` | `2` | 每个子代最多替换的邻居数 nr |
| `--mutation-prob` | `0.3` | 变异概率 |
| `--mutation-ops-min` | `2` | 每次变异最少边操作次数 |
| `--mutation-ops-max` | `6` | 每次变异最多边操作次数 |

### 约束
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--max-cycle` | `floor(sqrt(n_nodes))` | 禁止此长度及以下的环，允许更长的环 |
| `--max-parents` | 无限制 | 每个节点最大父节点数 |
| `--max-sdiff` | `15` | 结构对称差总和上限 |
| `--mdl-penalty` | `1.0` | MDL 惩罚项缩放 (1.0=标准BIC，越小惩罚越轻) |

### Batch 模式
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch` | 无 | 并行运行次数（设为 N 启用 batch 模式） |
| `--workers` | CPU 核心数 | 并行 worker 数量 |

### 其他
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--seed` | `42` | 随机种子 |
| `--output` | `./output` | 输出目录 |
| `--no-plot` | false | 跳过生成图表 |

## 常用示例

```bash
# 快速测试 (小规模)
python3 main.py --model asia --pop-size 21 --generations 50 --neighbors 5

# 正式运行 (Alarm 网络, 37节点)
python3 main.py --model alarm --pop-size 100 --generations 500 \
    --max-sdiff 50 --n-samples 5000 --max-parents 4

# 使用自定义 BIF 先验
python3 main.py --bif ./data/my_network.bif --generations 200

# 自定义环限制 (禁止 <=5 的环)
python3 main.py --model asia --max-cycle 5

# 轻惩罚 (鼓励更复杂的图)
python3 main.py --model asia --mdl-penalty 0.5
```

## Batch 模式详解

Batch 模式 (`--batch N`)：

1. 预生成**共享先验网络**（原始图变异 6 次，seed=9999）和**共享数据集**（seed=9999）
2. 并行启动 N 个独立进程，每个使用不同 seed (42..41+N) 运行 MOEA/D
3. 汇总所有运行结果，计算全局非支配前沿

```bash
# 30 次并行，8 worker
python3 main.py --model alarm --batch 30 --workers 8 \
    --pop-size 100 --generations 10000 --n-samples 5000 \
    --max-sdiff 50 --max-parents 4 --output ./output/batch_alarm
```

## 输出文件

### 单次运行 (`--output` 目录)

| 文件 | 说明 |
|------|------|
| `result.pkl` | 完整 MOEADResult 对象 |
| `pareto_front.csv` | Pareto 前沿解 (index,edges,cycles,max_cycle_len,mdl,sdiff) |
| `pareto_front.png` | Pareto 前沿散点图 (全部解 + 先验/原始网络标记) |
| `pareto_front_acyclic.png` | 仅无环解的 Pareto 前沿图 |
| `convergence.png` | Pareto 解数量收敛曲线 |
| `objective_convergence.png` | MDL 和 Sdiff 各自收敛曲线 |
| `best_graph.bif` | Sdiff=0 的最优图 (BIF 格式) |
| `network_*.png` | 三个代表解的网络结构图 |

### Batch 模式额外输出

| 文件 | 说明 |
|------|------|
| `shared/prior.pkl` | 共享先验网络 |
| `shared/data.npy` | 共享数据集 |
| `batch_<seed>/` | 各次运行的独立输出目录 |
| `combined_pareto.png` | 全部运行的 Pareto 前沿汇总图 |
| `combined_pareto_acyclic.png` | 全部运行的无环解 Pareto 前沿汇总图 |

## 编程接口

```python
from src.config import MOEADConfig
from src.prior import PriorNetwork
from src.moead import MOEAD
from src.visualize import plot_pareto_front

# 加载先验
prior_graph, node_names, n_states = PriorNetwork.from_pgmpy_model("asia")

# 生成数据
data, _, _ = PriorNetwork.generate_data("asia", n_samples=500, seed=42)

# 配置
config = MOEADConfig(
    n_nodes=len(node_names), n_states=n_states,
    max_forbidden_cycle=3, max_symmetric_diff=15,
    n_weight_vectors=50, n_neighbors=10, n_generations=200,
    data=data, random_seed=42,
)

# 运行
moead = MOEAD(config, prior_graph, data, node_names)
result = moead.run()

# 查看结果
for g, f in zip(result.pareto_graphs, result.pareto_f):
    print(f"edges={len(g.get_edges())}, MDL={f[0]:.2f}, Sdiff={f[1]:.0f}")
```
