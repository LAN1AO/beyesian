# 学术调研："先搜后修"（Search-then-Repair）范式在贝叶斯网络结构学习中的应用

## 一、核心研究问题定义

本调研聚焦以下研究思路是否已有前人探索：

> **在贝叶斯网络（Bayesian Network, BN）结构学习中，在搜索阶段放松有向无环图（DAG）的无环约束（允许解中存在环），搜索结束后，再对输出结果进行最小代价的修复——删除或反转最少的边，使其成为合法的 DAG。**

该思路可形式化为以下两阶段框架：

- **阶段一（搜索/放松）**：在包含有环图（或部分有环图）的松弛空间中优化某个评分函数/目标函数，不强制 acyclicity 约束。
- **阶段二（修复/投影）**：将阶段一的输出图通过最小代价操作（删边、反转边）转化为 DAG。修复的代价最小化本质上是一个 **Feedback Arc Set (FAS)** 问题或其对偶 **Maximum Acyclic Subgraph (MAS)** 问题。

---

## 二、相关工作分类

经过系统性文献检索，将相关工作分为以下类别。每类注明是否属于"先搜后修"范式。

### 类别 A：明确属于"先搜后修"范式的工作

#### A1. Peng & Ding (2003) — 最早的明确"先搜后修"

- **引用信息**：Peng, H., & Ding, C. (2003). Structure search and stability enhancement of Bayesian networks. *Proceedings of the 3rd IEEE International Conference on Data Mining (ICDM 2003)*, 621–624.
- **方法概述**：三步法：
  1. 使用 K2+ 算法为每个节点搜索最优父集，生成候选有向图（O(n²)），**允许图中存在环**。
  2. 通过 SCC 检测找出所有环，使用贝叶斯似然损失最小准则决定删除哪些边（短环优先启发式）。
  3. 对被删边的受影响节点进行局部 K2+ 重搜修复结构，并引入 Edge Perturbation 进行稳定性增强。
- **与"先搜后修"的相似度**：**极高**。这正是"先搜后修"范式的早期实现——在搜索阶段不保证无环，然后通过最小化似然损失删除环边。
- **差异**：修复时使用的是局部似然损失而非全局最小 FAS；操作空间是 parent-set 空间。未提供 FAS 的近似比分析。

#### A2. Gillot & Parviainen (2021-2023) — OptiMAS/ProxiMAS：目前最完善的"先搜后修"

- **引用信息**：
  - Gillot, P., & Parviainen, P. (2022). Learning large DAGs by combining continuous optimization and feedback arc set heuristics. *PGM 2022*, PMLR 186, 157–168.
  - Gillot, P. (2023). *Scalable learning of Bayesian networks using feedback arc set-based heuristics* [Doctoral dissertation, University of Bergen].
- **方法概述**：
  - **搜索阶段**：使用无约束梯度下降优化目标函数（最小二乘 + L1 正则），**完全不施加 acyclicity 约束**，允许权重矩阵任意有环。
  - **修复阶段**：每次梯度更新后，将当前权重矩阵**投影到 DAG 空间**——求解加权 MAS 问题：保留权重绝对值最大的无环子图。使用 Eades, Lin & Smyth (1993) 的线性时间 O(m) 贪心 FAS 近似算法。
  - **迭代交替**：优化步 → 投影步，循环进行，纳入 **Online Convex Optimization (OCO)** 框架。
- **与"先搜后修"的相似度**：**极高**。核心思想明确为 "decoupling the optimization of the objective function from the acyclicity constraint"。
- **理论分析**：基于 OCO 框架的 regret bound O(√K)；warm-start 策略加速收敛。

#### A3. Varando (2020) — 无约束学习 + 后处理

- **引用信息**：Varando, G. (2020). Learning DAGs without imposing acyclicity. *arXiv:2006.03005*.
- **方法概述**：在高斯设定下，仅使用 L1 惩罚的稀疏矩阵分解（不施加 acyclicity 约束）进行优化，发现所得图在经验上几乎无环（near-acyclic），再使用后处理消除残存环。
- **与"先搜后修"的相似度**：**高**。明确质疑 acyclicity 约束的必要性，认为稀疏性自然导向无环。后处理细节较为简单（阈值化 + 删边）。

#### A4. Yin, Yu, Gao & Ji (2024) — DAG-NCMLP：投影框架下的非线性 DAG 学习

- **引用信息**：Yin, N., Yu, Y., Gao, T., & Ji, Q. (2024). Efficient nonlinear DAG learning under projection framework. *ICPR 2024*, LNCS 15306, 445–460.
- **方法概述**：先获得非无环图（无约束搜索），使用 DAG-NoCurl 框架将其**投影**到 DAG 等价空间中。
- **与"先搜后修"的相似度**：**高**。明确使用"projection"（投影）术语。

#### A5. Vandel, Mangin & de Givry (2012) — 局部搜索中的"临时环"

- **引用信息**：Vandel, J., Mangin, B., & de Givry, S. (2012). New local move operators for learning the structure of Bayesian networks. *PGM 2012*, 289–296.
- **方法概述**：在 score-based 随机贪心搜索中，引入 **SWAP\*** 算子：
  - 当普通 SWAP 操作会产生有向环时，SWAP\* **临时允许进入有环图空间**。
  - 在环空间中执行一系列操作（删除或替换边）来打破所有环并**恢复无环性**。
  - 整个过程保证最终 DAG 的评分高于原始解。
- **与"先搜后修"的相似度**：**极高**。这是"先搜后修"在离散局部搜索中的微观体现——在每个 move 内部临时放松无环约束。
- **差异**：修复发生在每个 move 的粒度（微尺度），而非整个搜索过程结束后（宏尺度）。

---

### 类别 B：连续优化中的"软约束/惩罚"方法

#### B1. NOTEARS (Zheng et al., 2018)

- 使用平滑矩阵指数 trace 函数 \( h(W) = \text{tr}(e^{W \circ W}) - d \) 作为 acyclicity 等式约束
- **不完全属于"先搜后修"**：acyclicity 作为硬等式约束在优化中强制执行。但 Wei et al. (2020) 证明 KKT 条件不可满足，实际上优化中常产生带环中间解，最终需阈值化+后处理
- **局限**：O(d³) 复杂度；数值不稳定

#### B2. GOLEM (Ng et al., 2020)

- 将 acyclicity 约束改为**软惩罚项**加在似然目标中，变为无约束优化
- **介于两者之间**：软惩罚 = 放松约束（允许环但惩罚），显式包含后处理消除残存环
- **优势**：优化更容易（无约束），可扩展到数千节点（GOLEM-EV）

#### B3. DAGMA (Bello et al., 2022)

- 使用日志行列式函数作为 acyclicity 表征，路径跟踪法（barrier method）逐步收紧约束
- **不完全属于"先搜后修"**：优化早期允许违反 acyclicity，随 μ→0 逐步收紧，但无显式"修复/投影"步骤

---

### 类别 C：Acyclicity 由构造保证的方法（非"先搜后修"，提供对比视角）

| 方法 | 论文 | 核心思想 |
|------|------|----------|
| **Order-based Search** | Park & Klabjan (2017), JMLR | 在节点拓扑序空间中搜索，天然保证无环 |
| **COSMO** | Massidda et al. (2024), ICLR | 可微 orientation matrix 构造性保证无环，O(d²) |
| **PC 算法** | Spirtes et al. (2000) | 条件独立性检验 → 骨架 → 定向，定向阶段天然产出 CPDAG |

#### C3. ExDAG (Rytíř et al., 2024) — 惰性约束生成

- 使用混合整数二次规划（MIQP）精确学习 DAG
- 使用**惰性约束回调**：只在当前整数解包含环时才添加环消除约束
- **关系**：整数规划中的"先搜后修"，可精确求解 50 节点规模

#### D2. Wei, Gao & Yu (2020) — NOTEARS 后处理

- 提出基于 KKT 分析的局部搜索后处理算法 NOTEARS-KKTS，改善 SHD 通常 2 倍以上
- **较高相似度**：核心贡献之一是"先运行 NOTEARS 得到可能不够好的 DAG，再后处理改进"

---

## 三、关键论文深度分析

### 论文 1：Peng & Ding (2003) — 先驱工作

| 维度 | 内容 |
|------|------|
| **场景** | Score-based（Bayesian score + K2 变体） |
| **搜索阶段允许的环** | 所有环 |
| **修复方法** | 短环优先 + 最小似然损失删边 + 局部重搜 |
| **理论分析** | 无 |
| **优势** | 最早提出，O(n²)，思路清晰 |
| **代价** | 局部贪心修复；删边不可逆；无理论保证 |

### 论文 2：Gillot & Parviainen — OptiMAS/ProxiMAS (2021-2023) — **最重要**

| 维度 | 内容 |
|------|------|
| **场景** | Continuous optimization (linear SEM) |
| **搜索阶段允许的环** | 所有环 |
| **修复方法** | 每次迭代用贪心 FAS 投影到 DAG；Eades 算法 O(m) |
| **理论分析** | OCO framework: regret bound O(√K) |
| **优势** | 扩展到数千节点；GPU 加速；避免 cubic acyclicity 函数 |
| **代价** | FAS 近似比影响解质量；无全局最优保证 |

### 论文 3：Vandel, Mangin & de Givry (2012) — 局部搜索视角

| 维度 | 内容 |
|------|------|
| **场景** | Score-based local search (discrete, BDeu score) |
| **搜索阶段允许的环** | SWAP\* 算子执行期间的临时环 |
| **修复方法** | Move 内部通过一系列操作打破所有环 |
| **理论分析** | 无 |

### 论文 4：GOLEM (Ng et al., 2020)

| 维度 | 内容 |
|------|------|
| **场景** | Continuous optimization (linear SEM, MLE) |
| **搜索阶段允许的环** | 惩罚而非禁止（无约束优化） |
| **修复方法** | 后处理启发式消除残存环 |
| **理论分析** | 似然函数优于 MSE；强凸条件下有收敛保证 |

---

## 四、与始终维护 DAG 约束的方法对比

| 维度 | 始终维护 DAG | 先搜后修 |
|------|-------------|----------|
| **搜索空间** | 限制在 DAG 子空间 | 可在更大的图空间中搜索 |
| **计算复杂度** | 每次 move/step 需检查 acyclicity O(d)~O(d³) | 搜索时 O(0)，修复时 O(m)~NP-hard |
| **解质量** | 保证最终输出合法 | 可能获得更高评分的 DAG（搜索更自由） |
| **理论保证** | 较完善（K2, GES 等） | FAS 近似比的理论分析尚不完全 |
| **可扩展性** | 受限于 acyclicity 维护 | 可扩展到更大问题 |
| **优化难度** | 约束优化 | 解耦为无约束优化 + 组合投影 |

---

## 五、总结

### 核心结论

1. **"先搜后修"这一研究思路已被前人从多个角度探索**，但**未被命名为统一的范式**。相关工作分散在离散局部搜索、连续优化、混合整数规划等不同技术路线中。

2. **最接近的完整现代实现**：Gillot & Parviainen 的 **OptiMAS/ProxiMAS** (2021-2023)，明确以 "decoupling optimization from acyclicity" 为核心，交替进行无约束梯度优化 + FAS 投影。

3. **与始终维护 DAG 的方法对比**，先搜后修的核心优势是**更大的搜索自由度**和**更好的可扩展性**，代价是 FAS 近似比影响解质量和理论保证的相对薄弱。

4. **与本项目的直接关联**：我们的 MOEA/D 搜索阶段放松环约束 → 最后 FAS 修复转 DAG，恰好是"先搜后修"范式在离散多目标进化算法中的**首次应用**。已有的 OptiMAS 限于连续优化，已有的 Peng & Ding 限于单目标贪心搜索。我们的工作在**多目标 + 进化算法**背景下具有新颖性。

## 参考文献

- Bello, K., Aragam, B., & Ravikumar, P. (2022). DAGMA: Learning DAGs via M-matrices and a log-determinant acyclicity characterization. *NeurIPS 2022*, 35.
- Dalakyan, A., & Pourahmadi, M. (2021). Learning Bayesian networks through Birkhoff polytope: A relaxation method. *arXiv:2107.01658*.
- Eades, P., Lin, X., & Smyth, W. F. (1993). A fast and effective heuristic for the feedback arc set problem. *Information Processing Letters*, 47(6), 319–323.
- Gillot, P. (2023). *Scalable learning of Bayesian networks using feedback arc set-based heuristics* [Doctoral dissertation, University of Bergen].
- Gillot, P., & Parviainen, P. (2022). Learning large DAGs by combining continuous optimization and feedback arc set heuristics. *PGM 2022*, PMLR 186, 157–168.
- Massidda, R., Landolfi, F., Cinquini, M., & Bacciu, D. (2024). Constraint-free structure learning with smooth acyclic orientations. *ICLR 2024*.
- Ng, I., Ghassami, A., & Zhang, K. (2020). On the role of sparsity and DAG constraints for learning linear DAGs. *NeurIPS 2020*, 33, 17943–17954.
- Peng, H., & Ding, C. (2003). Structure search and stability enhancement of Bayesian networks. *ICDM 2003*, 621–624.
- Rytíř, P., Wodecki, A., & Mareček, J. (2024). ExDAG: Exact learning of DAGs. *arXiv:2406.15229*.
- Vandel, J., Mangin, B., & de Givry, S. (2012). New local move operators for learning the structure of Bayesian networks. *PGM 2012*, 289–296.
- Varando, G. (2020). Learning DAGs without imposing acyclicity. *arXiv:2006.03005*.
- Wei, D., Gao, T., & Yu, Y. (2020). DAGs with no fears: A closer look at continuous optimization for learning Bayesian networks. *NeurIPS 2020*, 33, 3895–3906.
- Yin, N., Yu, Y., Gao, T., & Ji, Q. (2024). Efficient nonlinear DAG learning under projection framework. *ICPR 2024*, LNCS 15306, 445–460.
- Zheng, X., Aragam, B., Ravikumar, P., & Xing, E. P. (2018). DAGs with NO TEARS: Continuous optimization for structure learning. *NeurIPS 2018*, 31.
