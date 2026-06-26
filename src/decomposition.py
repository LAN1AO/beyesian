from __future__ import annotations

import numpy as np


def das_dennis_weights(n_objectives: int, n_divisions: int) -> np.ndarray:
    """生成 Das-Dennis 方法下的均匀权重向量。

    Args:
        n_objectives: 目标数（本项目固定为 2）
        n_divisions: 分区数 H，产生 C(H+n_obj-1, n_obj-1) 个向量

    Returns:
        (n_vectors, n_objectives) 权重矩阵
    """
    if n_objectives == 2:
        weights = np.zeros((n_divisions + 1, 2))
        for i in range(n_divisions + 1):
            weights[i, 0] = i / n_divisions
            weights[i, 1] = (n_divisions - i) / n_divisions
        return weights
    else:
        # 通用版本（以备后续扩展到更多目标）
        weights = []
        _das_dennis_recursive(n_objectives, n_divisions, [], weights)
        return np.array(weights)


def _das_dennis_recursive(n_obj, divisions, current, result):
    """递归生成 Das-Dennis 权重向量。"""
    if n_obj == 1:
        current.append(divisions)
        result.append([x / divisions for x in current])
        current.pop()
    else:
        for i in range(divisions + 1):
            current.append(i)
            _das_dennis_recursive(n_obj - 1, divisions - i, current, result)
            current.pop()


def chebyshev_aggregate(
    f_values: np.ndarray,
    weight: np.ndarray,
    ideal: np.ndarray,
    nadir: np.ndarray | None = None,
    eps: float = 1e-8,
    sdiff_alpha: float = 1.0,
) -> float:
    """归一化切比雪夫聚合函数。

    g(x | λ, z*, z^nad) = max_i { λ_i * |f_i(x) - z*_i| / (z^nad_i - z*_i) }

    当归一化分母为 0 时使用 eps 保护。

    Args:
        f_values: (n_objectives,) 目标函数值
        weight: (n_objectives,) 权重向量
        ideal: (n_objectives,) 理想点
        nadir: (n_objectives,) 最低点，若为 None 则不归一化
        eps: 防止除零

    Returns:
        聚合标量值（越小越好）
    """
    diff = np.abs(f_values - ideal)
    if nadir is not None:
        range_ = np.maximum(nadir - ideal, eps)
        diff = diff / range_
    scaled = weight * diff
    scaled[1] *= sdiff_alpha
    return float(np.max(scaled))


def compute_neighborhood(weights: np.ndarray, n_neighbors: int) -> np.ndarray:
    """为每个权重向量找到 T 个最近邻居（按欧氏距离）。

    Args:
        weights: (n_vectors, n_objectives) 权重矩阵
        n_neighbors: 邻居数量 T

    Returns:
        (n_vectors, T) 邻居索引矩阵，每行包含该权重向量的 T 个最近邻居索引
    """
    n_vectors = weights.shape[0]
    T = min(n_neighbors, n_vectors)
    distances = np.zeros((n_vectors, n_vectors))
    for i in range(n_vectors):
        distances[i] = np.sqrt(np.sum((weights - weights[i]) ** 2, axis=1))
    neighbors = np.argsort(distances, axis=1)[:, :T]
    return neighbors


def update_ideal(f_values: np.ndarray, ideal: np.ndarray) -> np.ndarray:
    """更新理想点: z*_i = min(z*_i, f_i)。"""
    return np.minimum(ideal, f_values)


def compute_nadir(population_f: np.ndarray) -> np.ndarray:
    """从种群目标值矩阵计算 nadir point（每个目标的最大值）。"""
    return np.max(population_f, axis=0)


def non_dominated_sort(f_values: np.ndarray) -> np.ndarray:
    """简单的非支配排序，返回 Pareto 前沿的布尔掩码。

    Args:
        f_values: (n_solutions, n_objectives) 目标值矩阵

    Returns:
        (n_solutions,) 布尔数组，True 表示在 Pareto 前沿上
    """
    n = f_values.shape[0]
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i: all f_j <= f_i and at least one f_j < f_i
            if np.all(f_values[j] <= f_values[i]) and np.any(
                f_values[j] < f_values[i]
            ):
                is_pareto[i] = False
                break
    return is_pareto
