# 多目标贝叶斯网络结构学习 — 使用指南

## 环境准备

```bash
cd beyesian
bash scripts/setup_venv.sh   # 一键创建虚拟环境
source venv/bin/activate
```

## 基础用法

```bash
# 1. 预生成实验数据（一次性）
python scripts/prepare_data.py

# 2. 单次运行 — 空图先验 + 500 样本
python3 main.py \
    --prior-file data/priors/asia_empty.pkl \
    --data-file data/synthetic/asia_N500.npy

# 3. 单次运行 — 带真实图评估 (输出 SHD/F1)
python3 main.py \
    --prior-file data/priors/alarm_empty.pkl \
    --data-file data/synthetic/alarm_N5000.npy \
    --ground-truth data/ground_truth/alarm_graph.pkl \
    --pop-size 100 --generations 500 --max-parents 4

# 4. 单次运行 — 部分先验 (50% 节点已知)
python3 main.py \
    --prior-file data/priors/alarm_pct050.pkl \
    --data-file data/synthetic/alarm_N5000.npy \
    --pop-size 100 --generations 500 --max-parents 4

# 5. Batch 并行 — 同一先验+数据，跑 20 次汇总
python3 main.py \
    --prior-file data/priors/alarm_empty.pkl \
    --data-file data/synthetic/alarm_N5000.npy \
    --batch 20 --workers 8 \
    --pop-size 100 --generations 500 --max-parents 4
```

## 参数说明

### 先验网络与数据 (必填)
| 参数 | 说明 |
|------|------|
| `--prior-file` | 预生成先验网络文件 (.pkl)，由 `scripts/prepare_data.py` 生成 |
| `--data-file` | 预生成数据文件 (.npy)，由 `scripts/prepare_data.py` 生成 |

### 可选
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--ground-truth` | 无 | 真实图文件 (.pkl)，标注 GT 位置并输出 SHD/F1 到 CSV |
| `--model` | prior 文件名 | 网络名称标签 (仅用于 `params.json` 和图表标题) |

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
| `--crossover-type` | `sequential` | 交叉算子: sequential (默认) / no-cycle-check |

### 约束
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--max-parents` | 无限制 | 每个节点最大父节点数 |
| `--max-sdiff` | 自动计算 | 结构对称差总和上限 (默认: n_known×K+E_prior，K=max_parents 或 n-1) |
| `--mdl-penalty` | `1.0` | MDL 惩罚项缩放 (1.0=标准BIC，越小惩罚越轻) |
| `--sdiff-alpha` | `1.0` | sdiff 项缩放 (先验置信度)，<1 削弱先验牵引，1.0=不缩放 |

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
| `--plot-networks` | false | 输出三个代表解的网络结构图 |

## 常用示例

```bash
# 快速测试 — Asia + 空图先验
python3 main.py \
    --prior-file data/priors/asia_empty.pkl \
    --data-file data/synthetic/asia_N500.npy \
    --pop-size 21 --generations 50 --neighbors 5

# 正式运行 — Alarm 完整先验
python3 main.py \
    --prior-file data/priors/alarm_pct100.pkl \
    --data-file data/synthetic/alarm_N5000.npy \
    --pop-size 100 --generations 500 --max-parents 4

# 部分先验 (30% 节点已知)
python3 main.py \
    --prior-file data/priors/alarm_pct030.pkl \
    --data-file data/synthetic/alarm_N5000.npy \
    --pop-size 100 --generations 500

# 带真实图评估 (输出 SHD/F1 指标)
python3 main.py \
    --prior-file data/priors/alarm_pct100.pkl \
    --data-file data/synthetic/alarm_N5000.npy \
    --ground-truth data/ground_truth/alarm_graph.pkl \
    --pop-size 100 --generations 500

# 轻惩罚 (鼓励更复杂的图)
python3 main.py \
    --prior-file data/priors/asia_pct100.pkl \
    --data-file data/synthetic/asia_N500.npy \
    --mdl-penalty 0.5
```

## Batch 模式详解

Batch 模式 (`--batch N`)：

1. 所有 worker 共享同一份先验网络和数据文件
2. 并行启动 N 个独立进程，每个使用不同 seed (42..41+N) 运行 MOEA/D
3. 汇总所有运行结果，计算全局非支配前沿

```bash
# 30 次并行，8 worker
python3 main.py \
    --prior-file data/priors/alarm_empty.pkl \
    --data-file data/synthetic/alarm_N5000.npy \
    --batch 30 --workers 8 \
    --pop-size 100 --generations 10000 --max-parents 4 \
    --output ./output/batch_alarm
```

## 批量实验矩阵 (实验台)

`scripts/run_experiment.py` 是配置文件驱动的通用实验台：对一个超参矩阵 `networks × priors × alphas × n_samples × seeds` 逐 cell 调用 `main.py`，断点续跑（cell 的 `result.pkl` 已存在则跳过），最后汇总每 cell 的 best-F1_skel 行成 `summary.csv`。所有实验维度与超参均来自一个 JSON 配置文件，命令行不接收实验参数。数据需先由 `scripts/prepare_data.py` 预生成。

```bash
python scripts/run_experiment.py configs/alpha_knob.json                # 全量(断点续跑)+ 汇总
python scripts/run_experiment.py configs/alpha_knob.json --summary-only # 仅汇总已有结果
```

配置文件格式 (JSON)：

```json
{
  "output": "output/exp_alpha",
  "workers": 8,
  "networks": ["alarm"],
  "priors": ["gt", "mild", "moderate", "severe", "random"],
  "alphas": [1.0, 0.5, 0.25, 0.1, 0.05, 0.01],
  "n_samples": [1000],
  "seeds": [42, 43, 44],
  "params": { "pop_size": 100, "max_parents": 6, "generations": 3000 },
  "per_network": { "andes": { "pop_size": 200, "max_parents": 8 } }
}
```

| 字段 | 说明 |
|------|------|
| `networks/priors/alphas/n_samples/seeds` | 矩阵五维（列表），笛卡尔积 = 全部实验 cell；单值列表即固定该维 |
| `alphas` | 透传 `--sdiff-alpha`（先验置信度，<1 削弱先验牵引） |
| `params` | 透传 `main.py` 的全局超参默认（`pop_size/max_parents/generations/neighbors/…`） |
| `per_network` | 可选，按网络覆盖部分 `params`（规模不同的网络用不同 `pop_size/max_parents`） |
| `output` / `workers` | 输出目录（必填）/ 并行数（默认 CPU 核心数） |

输出目录为 `{output}/{net}_{prior}_a{alpha}_N{n}/run_{seed}/`，汇总写入 `{output}/summary.csv`（列：`network,prior,alpha,n_samples,seed,n_pareto,edges,mdl,sdiff,shd,f1,shd_skel,f1_skel`）。完整字段约定见脚本 docstring。

开箱即用的样例配置（`configs/`）：

| 配置 | 实验矩阵 |
|------|----------|
| `alpha_knob.json` | alarm × 5 先验 × 6 档 α × N1000 × 3 seed —— α 作为先验置信度旋钮 |
| `alpha_datasize.json` | alarm × {random,severe,gt} × α{1.0,0.25,0.05} × N{500…10000} × 3 seed —— 数据量×α 交互 |
| `full.json` | 6 网络 × 5 先验 × α1.0 × N{500…10000} × 30 seed —— 全量主实验 |

## 输出文件

### 单次运行 (`--output` 目录)

| 文件 | 说明 |
|------|------|
| `result.pkl` | 完整 MOEADResult 对象 |
| `pareto_front.csv` | Pareto 前沿解 (index,edges,mdl,sdiff[,shd,f1],count) |
| `params.json` | 本次实验的完整参数记录 |
| `pareto_front.png` | Pareto 前沿散点图 (全部解 + 先验/原始网络标记) |
| `convergence.png` | Pareto 解数量收敛曲线 |
| `objective_convergence.png` | MDL 和 Sdiff 各自收敛曲线 |
| `network_*.png` | 三个代表解的网络结构图 (需 `--plot-networks`) |

### Batch 模式额外输出

| 文件 | 说明 |
|------|------|
| `batch_<seed>/` | 各次运行的独立输出目录 |
| `combined_pareto.png` | 全部运行的 Pareto 前沿汇总图 |
| `params.json` | 实验参数记录 |

## 编程接口

```python
import pickle, numpy as np
from src.config import MOEADConfig
from src.moead import MOEAD
from src.visualize import plot_pareto_front

# 加载先验网络和数据（预生成文件）
with open("data/priors/asia_empty.pkl", "rb") as f:
    obj = pickle.load(f)
prior_graph = obj["graph"]
node_names = obj["node_names"]
n_states = obj["n_states"]

data = np.load("data/synthetic/asia_N500.npy").astype(np.int32)

# 配置
config = MOEADConfig(
    n_nodes=len(node_names), n_states=n_states,
    max_symmetric_diff=50,
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
