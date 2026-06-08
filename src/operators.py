from __future__ import annotations

import random

import numpy as np

from src.graph import DirectedGraph


def _fix_cycles(graph: DirectedGraph, rng: random.Random, max_iters: int = 1000):
    """通过随机删边修复图中的环。

    每次找到一个环，随机删除环中的一条边，重复至无环或达到最大迭代次数。
    """
    for _ in range(max_iters):
        cycle_path = None
        for __, path in graph._iter_cycles():
            cycle_path = path
            break
        if cycle_path is None:
            return
        idx = rng.randrange(len(cycle_path))
        u = cycle_path[idx]
        v = cycle_path[(idx + 1) % len(cycle_path)]
        graph.adj[u, v] = 0


def crossover(
    parent1: DirectedGraph,
    parent2: DirectedGraph,
    mdl_score,
    sdiff_score,
    weight: np.ndarray,
    ideal: np.ndarray,
    nadir: np.ndarray,
    eps: float = 1e-8,
    rng: random.Random | None = None,
    parent1_scores: dict[int, tuple[float, float]] | None = None,
    parent2_scores: dict[int, tuple[float, float]] | None = None,
    score_cache: dict | None = None,
    crossover_type: str = "sequential",
) -> DirectedGraph:
    """逐节点父集重组交叉算子，支持三种模式。

    sequential (默认): 用归一化切比雪夫聚合比较两父代父集，选更优者。
    random: 逐节点随机选择父代1或父代2的父集（验证 Chebyshev 驱动是否有效）。
    no-cycle-check: 先构建图（不判环），后通过随机删边修复环
                   （判断当前算子的优势区间）。

    Args:
        parent1, parent2: 两个父代图
        mdl_score: MDLScore 实例（sequential/no-cycle-check 模式用于缓存未命中兜底）
        sdiff_score: StructuralDiffScore 实例
        weight: 当前子问题的权重向量 (n_objectives,)
        ideal: 当前 ideal point
        nadir: 当前 nadir point
        eps: 防止除零
        rng: 随机数生成器
        parent1_scores: 父代1的 {node: (mdl, sdiff)} 缓存（可选）
        parent2_scores: 父代2的 {node: (mdl, sdiff)} 缓存（可选）
        score_cache: 组合缓存 (node, frozenset(parents)) → (mdl, sdiff)（可选）
        crossover_type: "sequential" | "random" | "no-cycle-check"

    Returns:
        子代图
    """
    if rng is None:
        rng = random.Random()

    n_nodes = parent1.n_nodes
    max_parents = parent1.max_parents
    child = DirectedGraph(n_nodes, max_parents=max_parents)
    check_cycles = (crossover_type != "no-cycle-check")
    use_chebyshev = (crossover_type != "random")

    # 随机化节点处理顺序，避免对 parent1 的偏向
    node_order = list(range(n_nodes))
    rng.shuffle(node_order)

    # 归一化分母（仅 Chebyshev 模式需要）
    if use_chebyshev:
        range_ = np.maximum(nadir - ideal, eps)

    for node in node_order:
        p1_parents = parent1.get_parents(node)
        p2_parents = parent2.get_parents(node)

        if set(p1_parents) == set(p2_parents):
            selected = p1_parents
        elif not use_chebyshev:
            # random 模式：随机选择父集
            selected = p1_parents if rng.random() < 0.5 else p2_parents
        else:
            # Chebyshev 驱动选择（sequential / no-cycle-check）
            if parent1_scores is not None and node in parent1_scores:
                mdl1, sd1 = parent1_scores[node]
            else:
                key = (node, frozenset(p1_parents))
                if score_cache is not None and key in score_cache:
                    mdl1, sd1 = score_cache[key]
                else:
                    mdl1 = mdl_score.score_node(node, p1_parents)
                    sd1 = sdiff_score.score_node(node, p1_parents)
                    if score_cache is not None:
                        score_cache[key] = (mdl1, sd1)

            if parent2_scores is not None and node in parent2_scores:
                mdl2, sd2 = parent2_scores[node]
            else:
                key = (node, frozenset(p2_parents))
                if score_cache is not None and key in score_cache:
                    mdl2, sd2 = score_cache[key]
                else:
                    mdl2 = mdl_score.score_node(node, p2_parents)
                    sd2 = sdiff_score.score_node(node, p2_parents)
                    if score_cache is not None:
                        score_cache[key] = (mdl2, sd2)

            # 归一化切比雪夫聚合: g = max_i { λ_i * |f_i - z*_i| / range_i }
            f1 = np.array([mdl1, sd1])
            f2 = np.array([mdl2, sd2])
            g1 = np.max(weight * np.abs(f1 - ideal) / range_)
            g2 = np.max(weight * np.abs(f2 - ideal) / range_)

            selected = p1_parents if g1 <= g2 else p2_parents

        # 组装: 加入选定的父节点边
        for p in selected:
            if check_cycles:
                child.add_edge(p, node)
            else:
                if p != node and not child.adj[p, node]:
                    if max_parents is None or child.get_in_degree(node) < max_parents:
                        child.adj[p, node] = 1

    # no-cycle-check 模式：事后修复环
    if not check_cycles:
        _fix_cycles(child, rng)

    return child


def mutate(
    graph: DirectedGraph,
    config,
    rng: random.Random | None = None,
) -> DirectedGraph:
    """随机变异算子。

    对图执行 k 次随机边操作（k ∈ [mutation_ops_min, mutation_ops_max]），
    每次等概率选择加边、删边或反转边。

    Args:
        graph: 待变异的图
        config: MOEADConfig 实例
        rng: 随机数生成器

    Returns:
        变异后的新图（不修改原图）
    """
    if rng is None:
        rng = random.Random()

    g = graph.copy()
    n = g.n_nodes
    k = rng.randint(config.mutation_ops_min, config.mutation_ops_max)
    ops = ["add", "remove", "reverse"]
    max_retries = 20  # 每次操作的最大重试次数

    for _ in range(k):
        for __ in range(max_retries):
            op = rng.choice(ops)
            if op == "add":
                u = rng.randrange(n)
                v = rng.randrange(n)
                if not g.has_edge(u, v) and g.add_edge(u, v):
                    break
            elif op == "remove":
                edges = g.get_edges()
                if edges:
                    u, v = rng.choice(edges)
                    g.remove_edge(u, v)
                    break
            else:  # reverse
                edges = g.get_edges()
                if edges:
                    u, v = rng.choice(edges)
                    if g.reverse_edge(u, v):
                        break
        # 如果重试耗尽，跳过本次操作

    return g


def select_parents(
    pop_size: int,
    idx: int,
    neighbor_indices: np.ndarray,
    prob_neighbor: float,
    rng: random.Random | None = None,
) -> tuple[int, int]:
    """为个体 idx 选择两个父代。

    以 prob_neighbor 概率从邻居中选择，否则从全局种群中选择。

    Args:
        pop_size: 种群大小
        idx: 当前个体索引
        neighbor_indices: shape (pop_size, T) 的邻居索引矩阵
        prob_neighbor: δ: 从邻居选择的概率
        rng: 随机数生成器

    Returns:
        (parent1_idx, parent2_idx)
    """
    if rng is None:
        rng = random.Random()

    if rng.random() < prob_neighbor:
        # 从邻居中选两个不同的个体
        neighbors = neighbor_indices[idx].tolist()
        a, b = rng.sample(neighbors, 2)
    else:
        # 从全局种群中选两个不同的个体（不含 idx 自身）
        candidates = [i for i in range(pop_size) if i != idx]
        a, b = rng.sample(candidates, 2)

    return a, b
