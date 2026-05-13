# 线性 DAG 连续优化问题的基本原理

## 1. 背景：离散组合搜索 vs 连续优化

### 1.1 传统方法：离散组合搜索

传统的 score-based 贝叶斯网络结构学习是**离散组合优化**问题：

$$\min_{G \in \text{DAG}} S(G, D)$$

其中 $G$ 是 DAG，$S$ 是评分函数（BIC、BDeu 等）。搜索空间在所有 DAG 中，其数量随节点数以**超指数**增长（Robinson, 1977）：

$$|\text{DAG}(d)| \sim d! \cdot 2^{\binom{d}{2}}$$

传统方法（爬山法、K2、GES 等）在离散空间中做局部搜索，每次移动（加边/删边/反转边）必须检查是否引入环（$O(d)$），严重限制可扩展性。

### 1.2 核心问题

能否将离散的 DAG 结构学习转化为**连续优化**问题，利用成熟的梯度下降等方法高效求解？

---

## 2. 线性结构方程模型 (Linear SEM)

### 2.1 模型定义

假设数据由以下线性结构方程模型生成：

$$X_j = \sum_{i \in \text{Pa}(j)} W_{ij} X_i + \varepsilon_j$$

用矩阵形式表示：

$$X = X W + \varepsilon, \quad \text{或等价地} \quad X = \varepsilon (I - W)^{-1}$$

其中：
- $X \in \mathbb{R}^{m \times d}$：$m$ 个样本，$d$ 个变量
- $W \in \mathbb{R}^{d \times d}$：加权邻接矩阵，$W_{ij} \neq 0$ 表示存在边 $i \rightarrow j$
- $\varepsilon$：独立噪声项

**关键性质**：如果 $G(W)$ 是一个 DAG，则 $W$ 的所有特征值为零（$W$ 是幂零矩阵）。这意味着存在一个节点排序使 $W$ 可化为严格上三角矩阵。

### 2.2 从图结构到权重矩阵

在连续优化框架中，我们**直接优化权重矩阵 $W$**，而非离散的图结构。$W_{ij}$ 的绝对值大小代表边 $i \rightarrow j$ 的强度；$W_{ij} = 0$ 表示无边。

目标变为：

$$\min_{W \in \mathbb{R}^{d \times d}} \ell(W; X) \quad \text{s.t.} \quad G(W) \text{ is a DAG}$$

其中 $\ell$ 是损失函数。对于线性 SEM，典型的损失函数是**最小二乘**：

$$\ell(W; X) = \frac{1}{2m} \|X - XW\|_F^2$$

等价于对每个节点 $j$ 独立求解一个线性回归 $X_j \sim \mathbf{X}_{\text{Pa}(j)}$，加上稀疏正则化（L1 惩罚）鼓励 $W$ 稀疏：

$$\ell(W; X) = \frac{1}{2m} \|X - XW\|_F^2 + \lambda \|W\|_1$$

---

## 3. NOTEARS：平滑无环约束的突破

### 3.1 核心洞察

Zheng et al. (2018) 的核心贡献是发现了一个**可微函数 $h(W)$ 来刻画 DAG 的无环性**：

$$h(W) = \text{tr}(e^{W \circ W}) - d = 0 \quad \Longleftrightarrow \quad G(W) \text{ 是 DAG}$$

其中 $\circ$ 表示逐元素乘积（Hadamard product），$\text{tr}$ 是矩阵的迹（对角元素和），$e^{A}$ 是矩阵指数。

### 3.2 数学原理

**第一步：幂零性与迹的关系**

对于一个 DAG，其加权邻接矩阵 $W$ 的所有特征值均为零。考虑矩阵 $W \circ W$（逐元素平方，非负矩阵），它与 $W$ 有相同的零模式（哪些位置非零）。$W \circ W$ 也是幂零矩阵，其特征值也全为零。

矩阵的迹等于其特征值之和。对于任意方阵 $A$：
- $\text{tr}(A^k) = \sum_i \lambda_i^k$（特征值的 k 次幂之和）
- 矩阵指数：$e^A = \sum_{k=0}^{\infty} \frac{A^k}{k!}$

**第二步：为什么 $W \circ W$ 而非 $W$**

使用 $W \circ W$ 而非 $W$ 的原因：
1. $W$ 可能有负权重（回归系数可为负），$W \circ W$ 保证所有元素非负
2. 非负矩阵的矩阵指数 $\text{tr}(e^{A})$ 当且仅当 $A$ 没有非零特征值时等于 $d$（维度）

**第三步：关键等式**

$$\text{tr}(e^{W \circ W}) = \sum_{i=1}^d e^{\lambda_i}$$

其中 $\lambda_i$ 是 $W \circ W$ 的特征值。由于 $W \circ W$ 是非负矩阵，其特征值 $\lambda_i \geq 0$（Perron-Frobenius 定理）。因此：

- 若 $G(W)$ 是 DAG：$W \circ W$ 是幂零矩阵 → 所有 $\lambda_i = 0$ → $\text{tr}(e^{W \circ W}) = \sum_i e^0 = d$
- 若 $G(W)$ 含环：$W \circ W$ 有正特征值 → 某些 $\lambda_i > 0$ → $\text{tr}(e^{W \circ W}) > d$

因此 $h(W) = \text{tr}(e^{W \circ W}) - d = 0 \Longleftrightarrow$ DAG，且 $h(W) > 0 \Longleftrightarrow$ 含环。

### 3.3 梯度

$h(W)$ 的梯度具有封闭形式：

$$\nabla_W h(W) = 2 \left(e^{W \circ W}\right)^\top \circ W$$

这使得可以使用标准的梯度下降方法。复杂度为 $O(d^3)$（矩阵指数计算）。

### 3.4 完整优化问题

NOTEARS 求解：

$$\min_{W \in \mathbb{R}^{d \times d}} \frac{1}{2m} \|X - XW\|_F^2 + \lambda \|W\|_1 \quad \text{s.t.} \quad h(W) = 0$$

使用**增广拉格朗日法**（Augmented Lagrangian）：

$$\mathcal{L}_{\rho}(W, \alpha) = \ell(W) + \alpha h(W) + \frac{\rho}{2} h(W)^2$$

交替进行：
1. 固定 $\alpha$，用梯度下降最小化 $\mathcal{L}_{\rho}$ 关于 $W$
2. 更新拉格朗日乘子：$\alpha \leftarrow \alpha + \rho h(W)$
3. 必要时增大惩罚系数 $\rho$

---

## 4. 后续改进与变体

### 4.1 GOLEM (Ng et al., 2020)

**核心改进**：
1. 用**似然函数**替代最小二乘目标（更符合数据生成假设）：

   $$\ell(W; X) = \frac{1}{2} \sum_{i=1}^d \log\left(\|X_i - X \mathbf{W}_{:,i}\|^2\right) - \log|\det(I - W)|$$

2. 将硬等式约束 $h(W) = 0$ 改为**软惩罚项**：

   $$\min_W \ell(W; X) + \lambda_1 \|W\|_1 + \lambda_2 h(W)$$

   变为**无约束**优化，可使用标准优化器（Adam、L-BFGS）。

3. 理论证明：似然函数 + 软惩罚 + 适当正则化可**渐近恢复**真实 DAG（在 polytree 情形下）。

**GOLEM 明确包含后处理**：优化完成后消除残存环。

### 4.2 DAGMA (Bello et al., 2022)

**核心改进**：提出新的 acyclicity 函数：

$$h^s(W) = -\log\det(sI - W \circ W) + d \log s$$

其中 $s > 0$ 是标量参数。性质：
- $G(W)$ 是 DAG $\Longleftrightarrow$ $\lim_{s \to 0} h^s(W) = 0$
- 梯度计算比矩阵指数更快
- 使用**对数障碍法**（logarithmic barrier）替代增广拉格朗日，通过逐步减小 $s$ 收紧约束

### 4.3 NOTEARS-KKTS (Wei et al., 2020)

**关键发现**：证明 NOTEARS 原始形式的 KKT 最优性条件**实际上不可满足**——解释了 NOTEARS 输出常含残存环的原因。

提出基于 KKT 分析的**后处理**：局部搜索删除/反转边，通常将 SHD 改善 2 倍以上。这本质上也是一种"先搜后修"。

### 4.4 COSMO (Massidda et al., 2024, ICLR)

**完全不同的路线**：通过可微的 smooth orientation matrix（从优先级向量参数化）**构造性保证**输出始终无环。$O(d^2)$ 复杂度。不需要 acyclicity 约束或后处理。

---

## 5. 与"先搜后修"范式的关联

### 5.1 NOTEARS 系列的隐含"先搜后修"

NOTEARS/GOLEM/DAGMA 在实际运行中：
1. **搜索阶段**：增广拉格朗日的早期迭代中，$h(W)$ 约束未被完全满足，$W$ 常是带环的
2. **收紧阶段**：随惩罚系数增大，$h(W)$ 逐步逼近 0
3. **后处理**：阈值化删除小权重边 + 局部搜索修复残存环

因此 NOTEARS 虽声称是"硬约束"，但实际运行中存在隐含的 relax-then-repair 过程。

### 5.2 与 OptiMAS 的对比

| | NOTEARS/GOLEM | OptiMAS/ProxiMAS |
|---|---|---|
| **acyclicity 实施** | 连续约束函数 $h(W)$ | 离散 FAS/MAS 投影 |
| **复杂度** | $O(d^3)$（矩阵指数/行列式） | $O(m)$（Eades 贪心） |
| **搜索-修复分离** | 隐式（惩罚逐步收紧） | 显式（交替优化-投影） |
| **理论保证** | 渐近恢复（GOLEM） | OCO regret bound |
| **可扩展性** | 受限于 $O(d^3)$，通常 < 1000 节点 | 数千节点（GPU加速） |
| **输出质量** | 可能残留环，需后处理 | 每次投影保证 DAG |

---

## 6. 小结

线性 DAG 连续优化的核心发展脉络：

```
离散组合搜索（NP-hard）
    ↓ Zheng et al. (2018)
连续优化 + 平滑 acyclicity 函数 h(W)
    ↓ Ng et al. (2020)
软惩罚替代硬约束（GOLEM）
    ↓ Bello et al. (2022)
更高效的 acyclicity 函数 + 对数障碍法（DAGMA）
    ↓ Massidda et al. (2024)
参数化保证无环（COSMO，不再需要 h(W)）
```

另一条并行路线（与本项目更相关）：

```
无环约束被视为优化障碍
    ↓ Gillot & Parviainen (2021)
分离优化与 acyclicity → 交替无约束梯度 + FAS 投影
    ↓ (本项目)
多目标进化搜索 + FAS 修复 → Pareto 前沿 → DAG
```

## 参考文献

- Zheng, X., Aragam, B., Ravikumar, P., & Xing, E. P. (2018). DAGs with NO TEARS: Continuous optimization for structure learning. *NeurIPS 2018*, 31.
- Ng, I., Ghassami, A., & Zhang, K. (2020). On the role of sparsity and DAG constraints for learning linear DAGs. *NeurIPS 2020*, 33, 17943–17954.
- Bello, K., Aragam, B., & Ravikumar, P. (2022). DAGMA: Learning DAGs via M-matrices and a log-determinant acyclicity characterization. *NeurIPS 2022*, 35, 16906–16919.
- Wei, D., Gao, T., & Yu, Y. (2020). DAGs with no fears: A closer look at continuous optimization for learning Bayesian networks. *NeurIPS 2020*, 33, 3895–3906.
- Massidda, R., Landolfi, F., Cinquini, M., & Bacciu, D. (2024). Constraint-free structure learning with smooth acyclic orientations. *ICLR 2024*.
- Gillot, P., & Parviainen, P. (2022). Learning large DAGs by combining continuous optimization and feedback arc set heuristics. *PGM 2022*, PMLR 186, 157–168.
- Robinson, R. W. (1977). Counting unlabeled acyclic digraphs. In *Combinatorial Mathematics V* (pp. 28–43). Springer.
