# 先验牵引力分析：为什么结果总靠近先验，以及如何削弱

> 主题：多目标 MOEA/D（MDL + sdiff）中，最终解系统性靠近先验网络。本文给出文献定位、数据验证、机制诊断（含一个被实测证伪的初始假设）和削弱方案。结论性分析，暂未落地代码改动。

---

## 背景

双目标：`f = (MDL, sdiff)`，其中 **sdiff = 结果相对先验网络的结构对称差**（sdiff 越小 = 越贴先验）。先验由 GT 扰动构造，分 gt / mild / moderate / severe / random 五级。观察到的现象：**结果系统性出现在先验附近**，且大图上即使先验很差（random），数据也几乎纠正不回来。

---

## 一、文献定位："结果靠近先验"是软先验的正常行为

引入先验的结构学习里，"结果偏向先验"是**理论预期**，称先验作为软约束（soft constraint）：

- **CaMML（O'Donnell et al.）** 最贴合本项目：候选结构的描述长度里**显式含一项"偏离先验的编辑距离代价"**——和我们把 sdiff 当目标几乎同构。置信度参数 `P_d` 控制拉力：高置信→偏向先验，`P_d=0.5`→先验无影响。
- **Scutari (2017)《Beyond Uniform Priors》**：数据稀缺时结构强烈受先验影响，数据充足时数据主导。
- **Borboudakis & Tsamardinos (UAI 2013)**：正确先验降低 SHD，错误先验增加 SHD。
- **KG-SoftMAP (2026)**：置信度加权、可被数据覆盖的软先验，随先验质量优雅降级。

### 对比实验的四种标准范式

| 范式 | 做法 | 本项目 |
|------|------|:---:|
| ① 变先验质量 | gt→…→random，看结果质量随之单调变化 | ✅ |
| ② 变样本量 | 看先验影响是否随数据增多而减弱 | ✅ |
| ③ with-prior vs 无先验 baseline | 对比 HC/Tabu 证明先验增益 | ✅ |
| ④ 变先验强度/置信度 | 扫 `P_d`/惩罚权重那类参数 | ⚠️ 缺（见方向 1） |

---

## 二、数据验证：牵引真实存在，且随先验质量变化

基于全量 3600 次运行（6 网络 × 5 先验 × 4 样本量 × 30 seed）。脚本 `analyze_prior_hypothesis.py` 可复跑。

**A — 结果是否靠近先验？**（ratio = 结果离先验 sdiff / 先验边数，越小越贴）

ratio 中位数 **0.133**，**65% 配置 < 0.3**。且贴合度随先验质量递减——算法非盲目贴先验：

| 先验 | 典型 ratio | 解读 |
|------|:---:|------|
| gt | ≈0 | 几乎完全落在先验上 |
| mild | 0.002–0.25 | 紧贴 |
| severe | 0.02–1.09 | 开始背离 |
| random | 0.18–1.97 | 显著脱离 |

**B — 结果质量随先验质量单调变化？** 近乎确定性：

- `ρ(先验 shd_from_gt → 结果 shd_skel) = +0.99`，**24/24 组一致**
- `ρ(→ f1_skel) = −0.99`
- 两端：gt 先验 → F1≈0.999；random 先验 → F1≈0.267

**caveat（牵引过强的证据）**：大图差先验时数据纠错很弱——munin1 random 先验 SHD 526 → 结果 504（几乎困在先验骨架里）；小图能拉回——alarm random 先验 83 → 结果 53。

---

## 三、机制诊断：牵引来自切比雪夫的 `max` 平衡，**不是分母塌缩**

归一化切比雪夫（`decomposition.py:65-69` / `operators.py:114-115`）：

```
range_i = max(nadir_i − ideal_i, ε)          # 动态，每代从种群算
g(x) = max_i { λ_i · |f_i − ideal_i| / range_i }
```

### 被证伪的初始假设：range_sdiff 塌缩

直觉曾认为 `range_sdiff` 因贴先验初始化 + 正反馈而塌缩到接近 0，使 sdiff 方向过陡。**实测（alarm + random + N1000，800 代）推翻了它**：

| gen | range_mdl | range_sdiff | 比值 |
|----:|----------:|------------:|-----:|
| 0   | 5492 | 68 | 1.2e-2 |
| 20  | 2989 | 60 | 2.0e-2 |
| 100 | 4270 | 52 | 1.2e-2 |
| 799 | 7490 | 63 | 8.4e-3 |

`range_sdiff` **全程稳定在 ~60，从不塌缩**；变化的是 `range_mdl`（谷底 3000 → 回升 7500）。归一化后两维都在 [0,1]，量纲已被消除——"sdiff 绝对范围小"本身不构成问题。

### 真实机制

对一个 λ=(0.99, 0.01) 的偏 mdl 子问题，跟踪归一化距离 `d_i = |f_i − ideal_i| / range_i`：

| | 前期 | 收敛后 |
|---|---|---|
| `d_mdl` | 0.118 | **0.017**（分子缩小 + range_mdl 增大，双重下降） |
| `d_sdiff` | 0.609 | **1.010**（为追 mdl 牺牲结构，离先验越来越远，不降反升） |

切比雪夫取 `max(0.99·d_mdl, 0.01·d_sdiff)`，临界点 `d_sdiff/d_mdl = 99`。收敛后比值在 ~60–100 横跳，于是 **约 50% 的代（gen130 后 40–67%）sdiff 项反超成为 `max`**。

**根因**：切比雪夫的 `max` 结构**强制两维平衡**，不允许 mdl 无限好而放任 sdiff。一旦某子问题把 mdl 压到接近 ideal（`d_mdl→0`），无论 `λ_sdiff` 多小，只要 `d_sdiff` 不同步趋零，sdiff 项迟早反超成 bottleneck，算法转去压 sdiff（拉回先验）。`ideal_sdiff≈2`（贴先验）就是那个锚。**牵引不是一开始发生，而是每个偏 mdl 子问题压完 mdl 后被 sdiff 项"拽回"一部分**，最终 Pareto 解比纯 mdl 最优更靠先验。

> 一句话：方向判断对（sdiff 在偏 mdl 子问题里反客为主、主导聚合），但归因从"分母塌缩"修正为"mdl 归一化距离随收敛趋零、sdiff 归一化距离不降，二者交叉后 sdiff 反超"。

---

## 四、削弱方案

削弱牵引 = 降低 sdiff 项在聚合里的相对量级，让偏 mdl 子问题能专注压 mdl。

| 方向 | 做法 | 改动 | 利弊 |
|------|------|:---:|------|
| **1. 聚合层 α 衰减（推荐）** | sdiff 项乘相对权重 α<1（等价放大 range_sdiff） | 小（2 处 + 1 旋钮） | 对症；α 语义 = **先验置信度**，扫 α=1.0/0.5/0.25/0.1 顺带补齐范式④。风险：全局削弱，整条前沿可能整体远离先验 |
| 2. 权重层偏 mdl | Das-Dennis 权重分布偏 mdl | 小 | 不改聚合；但单个偏 mdl 子问题内部反超仍在，缓解不彻底 |
| 3. 问题层降级 | sdiff 由硬目标 → 约束 / mdl 正则项 | 大 | 最彻底；但放弃"双目标 MOEA/D"核心卖点 |

**推荐方向 1**：最小改动、对症，且把修复顺势变成论文的一个新实验维度（先验置信度 α）。

---

## 附：数据与脚本

- 全量结果：`output/experiments/summary.csv`
- 假设验证脚本：`analyze_prior_hypothesis.py`（分析 A/B 可复跑）
- 机制实测：alarm + random + N1000，临时打点 800 代（诊断后源码已还原）

**文献来源**：
- [Scutari 2017, Beyond Uniform Priors](https://ar5iv.labs.arxiv.org/html/1704.03942)
- [Constantinou et al. 2023, impact of prior knowledge](https://link.springer.com/article/10.1007/s10115-023-01858-x)
- [Chen et al. 2025, Mitigating Prior Errors (IEEE TPAMI)](https://ieeexplore.ieee.org/document/11106743)
- [Xu et al. 2026, KG-SoftMAP](https://browse-export.arxiv.org/abs/2606.10358)
- [Borboudakis & Tsamardinos, UAI 2013](https://www.stats.ox.ac.uk/~evans/uai13/Borboudakis.pdf)
- [CaMML technical report, Monash](https://au-east.erc.monash.edu.au/fpfiles/36404490/tr2006194full.pdf)
- [Niculescu-Mizil & Caruana, AISTATS 2007](https://www.cs.cornell.edu/~caruana/niculescu.mtlbnets.aistats07.pdf)
