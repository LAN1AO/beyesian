from dataclasses import dataclass, field
import numpy as np


@dataclass
class MOEADConfig:
    """MOEA/D 多目标贝叶斯网络结构学习配置。"""

    # --- 问题维度 ---
    n_nodes: int
    n_states: list[int]  # 每个节点的取值数 r_i

    # MDL 惩罚缩放（1.0 = 标准 BIC, 越小惩罚越轻）
    mdl_penalty_scale: float = 1.0

    # 最大父节点数限制（None = 不限制）
    max_parents: int | None = None

    # --- 搜索空间约束 ---
    max_symmetric_diff: int = 10  # 结构对称差总和上限

    # --- MOEA/D 参数 ---
    n_weight_vectors: int = 50  # 权重向量数 (种群大小)
    n_neighbors: int = 10  # 邻居数量 T
    n_generations: int = 200
    prob_neighbor_mating: float = 0.9  # δ: 从邻居选父代的概率
    max_replacements: int = 2  # nr: 每个子代最多替换邻居数

    # --- 遗传算子 ---
    crossover_prob: float = 0.9
    mutation_prob: float = 0.3
    mutation_ops_min: int = 2
    mutation_ops_max: int = 6

    # --- 数据 ---
    data: np.ndarray | None = None  # (m, n_nodes) 数据集

    # --- 输出 ---
    output_dir: str = "./output"
    random_seed: int = 42

    # --- 切比雪夫归一化 epsilon ---
    eps: float = 1e-8

    def __post_init__(self):
        if self.data is not None:
            m, n = self.data.shape
            if n != self.n_nodes:
                raise ValueError(f"data 列数 ({n}) 与 n_nodes ({self.n_nodes}) 不匹配")
