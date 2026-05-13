# 有向图最小评分损失转 DAG — 学术调研

## 1. 问题形式化

给定 MOEA/D 输出的有向图 $G=(V,E)$（可能包含长度 > n 的环），目标是将 $G$ 转化为 DAG $G'$，使得评分损失最小。

两个目标函数均具有**可分解性**（这是本问题的关键优势）：

$$S(G) = \sum_{i=1}^{n} s_i(\mathbf{Pa}_i)$$

MDL 和结构对称差 $\sigma$ 都仅依赖于每个节点的父节点集合。这意味着：**删除边 $u \rightarrow v$，仅需重算节点 $v$ 的局部评分**，其余 $n-1$ 个节点评分不变。

由此可以为每条边计算精确的"删除代价"：

$$w(u \rightarrow v) = s_v(\mathbf{Pa}_v \setminus \{u\}) - s_v(\mathbf{Pa}_v)$$

---

## 2. 核心算法范式

学术界的解法收敛到同一个框架：**Maximum Acyclic Subgraph (MAS) / Minimum Feedback Arc Set (FAS)**。

### 2.1 范式定义

| 概念 | 定义 |
|------|------|
| **MAS** | 给定边加权有向图，求总权重最大的**无环**子图 |
| **FAS** | 求总权重最小的边集，删除后图变无环 |
| **关系** | $\text{MAS} = \text{所有边} - \text{FAS}$：二者互逆 |

有向图转 DAG 的经典归约：

$$\text{含环图 } G \xrightarrow{\text{删除 FAS 边集}} \text{DAG } G'$$

### 2.2 关键论文谱系

**Gillot & Parviainen 系列（2020-2022）** — 最系统的 MAS/FAS + BN 学习研究：

| 论文 | 会议/期刊 | 贡献 |
|------|-----------|------|
| *Scalable BN Structure Learning via MAS* (2020) | PMLR 138 | 首次将 BN 结构学习的无环化步骤形式化为 MAS 问题，用 ILP 精确求解 |
| *Learning Large DAGs by Combining Continuous Optimization and FAS Heuristics* (2021) | AAAI 2022 | **OptiMAS / ProxiMAS**：梯度优化 + FAS 投影交替迭代，向量化 Eades 贪心，扩展到 20K 节点 |
| *Convergence of FAS-Based Heuristics* (2022) | PMLR 186 | 给出 FAS 投影 + 交替优化的**收敛性数学证明** |

**Gao, Yan, Wang & Liu (2023)** — 与本项目最直接相关：

> *"Bayesian Network Structure Learning Algorithm Based on Score Increment and Reduction"* — IEEE ICCRE 2023

核心思路：
1. 维护 **profit table**（每条可能添加边的评分增益）
2. 维护 **loss table**（每条已有边的评分损失，即删除该边的精确代价）
3. 贪心加最优边 → DFS 检测所有环 → 用 loss table 指导**删除环中代价最小的边**
4. 迭代直到 profit table 耗尽

**这是唯一显式利用评分可分解性构建精确 loss table 来指导破环的论文**，与本项目场景高度吻合。

---

## 3. Eades-Lin-Smyth 贪心 FAS 算法

**论文**: Eades, P., Lin, X. & Smyth, W.F. (1993). *A Fast and Effective Heuristic for the Feedback Arc Set Problem.* Information Processing Letters, 47(6):319–323.

### 3.1 算法特性

| 属性 | 值 |
|------|-----|
| 时间复杂度 | $O(m)$ — 线性时间 |
| 近似比 | $\frac{m}{2} - \frac{c_1 m}{\Delta^{1/2}}$ (Berger-Shor bound) |
| 类型 | 贪心近似启发式 |

### 3.2 算法步骤

1. 持续从图中移除节点：
   - 若节点无入边（source）→ 从左侧加入排序
   - 若节点无出边（sink）→ 从右侧加入排序
   - 否则 → 选 $|\text{out-degree} - \text{in-degree}|$ 最大的节点移除
2. 最终排序中，所有**反向边**（从右侧指向左侧）构成 FAS
3. **加权版本**：将度数替换为边权和

### 3.3 伪代码

```
算法: WeightedEadesFAS(G, w)
输入: 有向图 G=(V,E)，边权重函数 w(e)
输出: 待删除边集合 F

S = []   # 拓扑排序序列
while V 非空:
    while 存在无入边节点 v:
        S.prepend(v); 从 G 移除 v
    while 存在无出边节点 v:
        S.append(v); 从 G 移除 v
    if V 非空:
        选 v = argmax |sum_out_weight(v) - sum_in_weight(v)|
        S.prepend(v) if 出权 > 入权 else S.append(v)
        从 G 移除 v

F = { (u,v) ∈ E | u 在 S 中排在 v 之后 }
return F
```

---

## 4. 三种可行方案对比

### 方案 A：加权 FAS + Eades 贪心启发式（**推荐首选**）

**原理**：每条边赋权 $w(u \rightarrow v)$ = 精确的删除代价，用 Eades 贪心算法求近似最小 FAS。

**在双目标场景下**：将 $(\Delta\text{MDL}, \Delta\text{Sdiff})$ 标量化为单一权重。使用该解的切比雪夫权重向量做聚合：

$$w(e) = \max\left(\lambda_0 \cdot \frac{|\Delta\text{MDL}|}{\text{range}_0}, \lambda_1 \cdot \frac{|\Delta\text{Sdiff}|}{\text{range}_1}\right)$$

**优点**：线性时间、数学性质好、被 Gillot 系列验证可扩展至万级节点
**缺点**：贪心近似（无最优性保证，但实践中 FAS 占比很小）

### 方案 B：DFS 环检测 + loss table 贪心删边

**原理**（借鉴 Gao et al. 2023）：
1. DFS 找到图中所有环
2. 对每个环，选择环中**删除代价最小**的边删除
3. 重复直到无环

**改进**：不按"逐个环"处理，而是收集所有环的候选边，用贪心集合覆盖的方式选边。

**优点**：直接、利用精确评分损失、每次决策可解释
**缺点**：按什么顺序处理多个环会影响最终结果；环可能重叠；最坏情况指数级环数

### 方案 C：边反转 + 局部搜索

**原理**：不限于纯删除。对环中的每条边，同时考虑三种操作：
- **删除** $u \rightarrow v$：代价 $w_{\text{del}}$
- **反转** $u \rightarrow v$ 为 $v \rightarrow u$：代价 $w_{\text{rev}} = s_v(\mathbf{Pa}_v \setminus \{u\}) + s_u(\mathbf{Pa}_u \cup \{v\})$
- **不操作**

**优点**：可能保留更多结构信息；边反转在结构对称差度量下可能优于删边
**缺点**：搜索空间大、需持续检查环约束

---

## 5. 针对本项目的推荐方案

结合项目特点（评分可分解、双目标 Pareto 前沿），推荐**方案 A 为主 + 方案 C 为辅**：

```
算法：ParetoFront2DAG
输入：Pareto 前沿 {(G_k, λ_k)}，其中 λ_k 是该解的切比雪夫权重向量
输出：DAG 集合

对每个 Pareto 解 (G_k, λ_k):
  1. 为 G_k 的每条边计算权重：
     w(u→v) = chebyshev((ΔMDL_v, ΔSdiff_v), λ_k, ideal, nadir)
     （Δ 值通过 score_node 精确计算，O(1) 每条边）

  2. 调用 Eades 加权 FAS 算法，得到待删边集合 F

  3. 对 F 中每条边尝试反转（替代删除）：
     - 若反转不引入被禁短环 → 保留反转
     - 否则 → 删除

  4. 返回结果 DAG (G_k', 新的目标向量)
```

**复杂度分析**：对每个 Pareto 解，$O(m)$ 计算边权 + $O(m)$ 运行 Eades + $O(|F| \cdot d^3)$ 尝试反转。总体在毫秒级。

---

## 6. 其他相关文献

| 论文 | 要点 |
|------|------|
| **Park & Klabjan (2017)** — *BN Learning via Topological Order*, JMLR 18 | GD 算法：梯度下降 + MAS 投影的前驱 |
| **Ng, Ghassami & Zhang (2020)** — *GOLEM*, NeurIPS 2020 | 软 DAG 约束替代硬约束，Gillot 系列的对比基准 |
| **Zheng et al. (2018)** — *NOTEARS*, NeurIPS 2018 | 连续 DAG 约束 $h(W) = \text{tr}(e^{W \circ W}) - d$ |
| **Sun et al. (2017)** — *Breaking Cycles in Noisy Hierarchies*, WebSci 2017 | 多层级的破环方法对比（PageRank/TrueSkill/Ensemble） |
| **Vandel, Mangin & de Givry (2012)** — *New Local Move Operators for BN*, PGM 2012 | 临时允许环搜索 + 后处理无环化的算子设计 |

---

## 7. 核心参考文献

1. **Gillot, P. & Parviainen, P.** (2021). *Learning Large DAGs by Combining Continuous Optimization and Feedback Arc Set Heuristics.* AAAI 2022. [arXiv:2107.00571](http://www.arxiv.org/abs/2107.00571)

2. **Gillot, P. & Parviainen, P.** (2022). *Convergence of Feedback Arc Set-Based Heuristics for Linear Structural Equation Models.* PMLR 186:157–168.

3. **Gao, Y., Yan, X., Wang, Z. & Liu, X.** (2023). *Bayesian Network Structure Learning Algorithm Based on Score Increment and Reduction.* IEEE ICCRE 2023.

4. **Eades, P., Lin, X. & Smyth, W.F.** (1993). *A Fast and Effective Heuristic for the Feedback Arc Set Problem.* Information Processing Letters, 47(6):319–323.

5. **Gillot, P. & Parviainen, P.** (2020). *Scalable Bayesian Network Structure Learning via Maximum Acyclic Subgraph.* PMLR 138:209–220.

---

## 8. 总结

学术界的共识方案是：**利用评分可分解性计算精确边权 → 求解加权 FAS/MAS → 删除/反转最小代价边集**。

- **Gao et al. (2023)** 的 loss table 方法与我们的匹配度最高（同样是精确的可分解评分）
- **Gillot & Parviainen** 的 Eades 向量化 FAS 在可扩展性上最优
- 建议实现路径：先实现方案 A（加权 FAS），再可选加入方案 C 的边反转优化
