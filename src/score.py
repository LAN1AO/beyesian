from __future__ import annotations

import numpy as np


class MDLScore:
    """贝叶斯网络 MDL (Minimum Description Length) 评分。

    具有可分解性: 每个节点的评分仅依赖于其父节点集合。

    MDL(G, D) = Σ_i [ Σ_j Σ_k m_ijk * log2(m_ijk / m_ij)
                     - penalty_scale * 0.5 * q_i * (r_i - 1) * log2(m) ]

    penalty_scale 控制惩罚强度:
      1.0 = 标准 BIC/MDL
      0.5 = 半惩罚
      0.0 = 纯似然（无惩罚，必定会过拟合为完全图）
    """

    def __init__(self, data: np.ndarray, n_states: list[int],
                 penalty_scale: float = 1.0):
        """
        Args:
            data: (m_samples, n_nodes) 数据集，每列为整数编码 [0..r_i-1]
            n_states: 每个节点的取值数 r_i
            penalty_scale: 惩罚项缩放因子，默认 1.0（标准 BIC）
        """
        self.data = data.astype(np.int32)
        self.n_states = n_states
        self.n_nodes = len(n_states)
        self.n_samples = data.shape[0]
        self._log2_m = np.log2(self.n_samples)
        self._penalty_scale = penalty_scale

    def score_node(self, node: int, parents: list[int]) -> float:
        """计算单个节点在给定父集下的 MDL 评分分量。"""
        parent_configs = self._encode_parent_configs(node, parents)
        return self._score_from_configs(node, parents, parent_configs)

    def score_graph(self, graph) -> float:
        """计算整个图的 MDL 评分。"""
        total = 0.0
        for node in range(self.n_nodes):
            parents = graph.get_parents(node)
            total += self.score_node(node, parents)
        return total

    def score_nodes(self, graph, nodes: list[int]) -> dict[int, float]:
        """仅计算指定节点的 MDL 评分。"""
        result = {}
        for node in nodes:
            parents = graph.get_parents(node)
            result[node] = self.score_node(node, parents)
        return result

    def _encode_parent_configs(
        self, node: int, parents: list[int]
    ) -> np.ndarray:
        """将父节点取值组合编码为单个整数索引。

        config = Σ_{p} state[p] * stride[p]
        strides: stride[0]=1, stride[k]=stride[k-1]*r_{parent[k-1]}
        """
        n = self.n_samples
        if not parents:
            return np.zeros(n, dtype=np.int32)

        strides = [1]
        for p in parents:
            strides.append(strides[-1] * self.n_states[p])
        parent_strides = strides[:-1]  # 每个父节点的 stride

        configs = np.zeros(n, dtype=np.int32)
        for j, p in enumerate(parents):
            configs += self.data[:, p] * parent_strides[j]
        return configs

    def _score_from_configs(
        self, node: int, parents: list[int], parent_configs: np.ndarray
    ) -> float:
        """给定父节点配置编码，计算节点的 MDL 评分。"""
        r_i = self.n_states[node]
        total_parent_states = 1
        for p in parents:
            total_parent_states *= self.n_states[p]

        # m_ij: 每种父配置的样本数
        m_ij = np.bincount(parent_configs, minlength=total_parent_states)

        # joint_configs: (parent_config, node_value) 联合编码
        joint = parent_configs * r_i + self.data[:, node]
        m_ijk = np.bincount(joint, minlength=total_parent_states * r_i)

        # 似然项: Σ_j Σ_k m_ijk * log2(m_ijk / m_ij)  — 向量化版本
        m_ijk_2d = m_ijk.reshape(total_parent_states, r_i)
        # 仅计算 m_ijk > 0 的位置
        valid = m_ijk_2d > 0
        # m_ij 广播到 (total_parent_states, r_i)
        m_ij_bc = np.broadcast_to(m_ij[:, np.newaxis], (total_parent_states, r_i))
        mdl = np.sum(m_ijk_2d[valid] * np.log2(m_ijk_2d[valid] / m_ij_bc[valid]))

        # 惩罚项
        penalty = self._penalty_scale * 0.5 * total_parent_states * (r_i - 1) * self._log2_m
        return mdl - penalty


class StructuralDiffScore:
    """相较先验网络的结构对称差评分。

    σ_i = |Π_i(B_s) ∪ Π_i(B_sc)| - |Π_i(B_s) ∩ Π_i(B_sc)|
    σ   = Σ_i σ_i
    """

    def __init__(self, prior_graph):
        """先验网络的父节点集合会被预计算并缓存。"""
        self.n_nodes = prior_graph.n_nodes
        self._prior_parents: list[set[int]] = [
            set(prior_graph.get_parents(i)) for i in range(self.n_nodes)
        ]

    def score_node(self, node: int, parents: list[int]) -> float:
        """计算单个节点的结构对称差。"""
        p_candidate = set(parents)
        p_prior = self._prior_parents[node]
        union_size = len(p_candidate | p_prior)
        intersect_size = len(p_candidate & p_prior)
        return float(union_size - intersect_size)

    def score_graph(self, graph) -> float:
        """计算整个图的结构对称差总和。"""
        total = 0.0
        for node in range(self.n_nodes):
            parents = graph.get_parents(node)
            total += self.score_node(node, parents)
        return total

    def score_nodes(self, graph, nodes: list[int]) -> dict[int, float]:
        """仅计算指定节点的结构对称差。"""
        result = {}
        for node in nodes:
            parents = graph.get_parents(node)
            result[node] = self.score_node(node, parents)
        return result

    def get_prior_parents(self, node: int) -> set[int]:
        """返回先验网络中某节点的父节点集合。"""
        return self._prior_parents[node]
