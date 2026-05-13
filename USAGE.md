# 多目标贝叶斯网络结构学习 — 使用指南

## 环境准备

```bash
cd beyesian
pip install -r requirements.txt --break-system-packages
```

## 基础用法

```bash
# 使用 Asia 网络 (8节点)，自动生成合成数据
python3 main.py --model asia

# 指定样本数和世代数
python3 main.py --model asia --n-samples 1000 --generations 300 --pop-size 50
```

## 参数说明

### 先验网络
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `asia` | bnlearn 网络名，可选 `alarm`, `sachs`, `child`, `insurance` 等 |
| `--bif` | 无 | BIF 文件路径，优先级高于 `--model` |

### 数据
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--n-samples` | `500` | 合成数据的样本数 |
| `--data-file` | 无 | 外部数据 (.npy)，优先级高于合成数据 |

### MOEA/D 核心
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--pop-size` | `50` | 种群大小 / 权重向量数 |
| `--neighbors` | `10` | 邻居数量 T |
| `--generations` | `200` | 最大世代数 |
| `--prob-neighbor` | `0.9` | 从邻居选父代的概率 δ |
| `--max-replace` | `2` | 每个子代最多替换的邻居数 nr |
| `--mutation-prob` | `0.3` | 变异概率 |

### 约束
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--max-cycle` | `3` | 禁止此长度及以下的环 (2/3/4)；更长的环被允许，留待后续 DAG 转化处理 |
| `--max-parents` | 无限制 | 每个节点最大父节点数 |
| `--max-sdiff` | `15` | 结构对称差总和上限 |

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
    --max-sdiff 50 --n-samples 2000

# 使用自定义 BIF 先验
python3 main.py --bif ./data/my_network.bif --generations 200

# 仅禁止 2-环 (最宽松: 3+ 环均允许)
python3 main.py --model asia --max-cycle 2

# 禁止 ≤4 的环 (最严格: 仅允许 5+ 环)
python3 main.py --model asia --max-cycle 4 --max-sdiff 30
```

## 输出文件

运行后在 `--output` 目录下生成:

| 文件 | 说明 |
|------|------|
| `result.pkl` | 完整 MOEADResult 对象 (可 pickle 加载) |
| `pareto_front.csv` | Pareto 前沿解列表 (index,edges,mdl,sdiff) |
| `pareto_front.png` | Pareto 前沿散点图 |
| `convergence.png` | Pareto 解数量收敛曲线 |
| `objective_convergence.png` | MDL 和 Sdiff 各自收敛曲线 |
| `best_graph.bif` | Sdiff=0 的最优图 (BIF 格式) |

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

# 获取图的边列表
print(result.pareto_graphs[0].get_edges())
print(result.pareto_graphs[0].get_parents(3))
```
