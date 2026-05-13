from __future__ import annotations

import random

from src.graph import DirectedGraph


def generate_initial_population(
    prior_graph: DirectedGraph,
    sdiff_score,
    config,
    rng: random.Random | None = None,
) -> list[DirectedGraph]:
    """生成初始种群。

    从先验网络出发，施加不同强度的随机扰动（加边/删边/反转边），
    扰动次数随个体索引递增以保证种群多样性。
    每个个体都满足环约束和结构对称差约束。

    Args:
        prior_graph: 先验网络
        sdiff_score: StructuralDiffScore 实例，用于校验对称差
        config: MOEADConfig 实例
        rng: 随机数生成器

    Returns:
        长度为 n_weight_vectors 的图列表
    """
    if rng is None:
        rng = random.Random()

    n_pop = config.n_weight_vectors
    n_nodes = prior_graph.n_nodes
    max_sdiff = config.max_symmetric_diff
    population = []

    for i in range(n_pop):
        # 扰动强度随索引递增: 从 0 到 max_perturbation
        max_perturb = max(1, int(max_sdiff * (i + 1) / n_pop))
        graph = _generate_one(
            prior_graph,
            n_nodes,
            max_sdiff,
            max_perturb,
            sdiff_score,
            rng,
        )
        population.append(graph)

    return population


def _generate_one(
    prior_graph: DirectedGraph,
    n_nodes: int,
    max_sdiff: int,
    max_perturb: int,
    sdiff_score,
    rng: random.Random,
) -> DirectedGraph:
    """生成一个满足约束的随机图。

    策略: 从先验网络出发，施加 n_perturb 次随机边操作。
    若结果超出对称差约束，回退重试。
    """
    max_attempts = 50
    ops = ["add", "remove", "reverse"]

    for _ in range(max_attempts):
        g = prior_graph.copy()
        n_perturb = rng.randint(0, max_perturb)

        for __ in range(n_perturb):
            for ___ in range(20):  # 每次操作重试
                op = rng.choice(ops)
                if op == "add":
                    u = rng.randrange(n_nodes)
                    v = rng.randrange(n_nodes)
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

        # 校验对称差约束
        total_sdiff = sdiff_score.score_graph(g)
        if total_sdiff <= max_sdiff:
            return g

    # 若始终无法满足约束，返回先验网络本身
    return prior_graph.copy()
