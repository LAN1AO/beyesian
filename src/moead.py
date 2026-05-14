from __future__ import annotations

import random
import time

import numpy as np

from src.decomposition import (
    chebyshev_aggregate,
    compute_nadir,
    compute_neighborhood,
    das_dennis_weights,
    non_dominated_sort,
    update_ideal,
)
from src.graph import DirectedGraph
from src.operators import crossover, mutate, select_parents
from src.population import generate_initial_population
from src.score import MDLScore, StructuralDiffScore
from src.score_cache import CompositeScore


class MOEADResult:
    """MOEA/D 运行结果。"""

    def __init__(
        self,
        pareto_graphs: list[DirectedGraph],
        pareto_f: np.ndarray,
        population: list[DirectedGraph],
        population_f: np.ndarray,
        history: list[np.ndarray],
        ideal: np.ndarray,
        config,
        node_names: list[str],
        runtime: float,
    ):
        self.pareto_graphs = pareto_graphs
        self.pareto_f = pareto_f
        self.population = population
        self.population_f = population_f
        self.history = history
        self.ideal = ideal
        self.config = config
        self.node_names = node_names
        self.runtime = runtime

    def __repr__(self) -> str:
        return (
            f"MOEADResult(n_pareto={len(self.pareto_graphs)}, "
            f"runtime={self.runtime:.1f}s)"
        )


class MOEAD:
    """基于分解的多目标进化算法 (MOEA/D)。

    使用切比雪夫方法进行目标聚合，求解贝叶斯网络结构学习的
    双目标优化问题（MDL 评分 + 结构对称差）。
    """

    def __init__(
        self,
        config,
        prior_graph: DirectedGraph,
        data: np.ndarray,
        node_names: list[str] | None = None,
    ):
        self.config = config
        self.prior_graph = prior_graph
        self.n_nodes = prior_graph.n_nodes

        if node_names is None:
            node_names = [str(i) for i in range(self.n_nodes)]
        self.node_names = node_names

        # 将 max_parents 注入先验图，后续 copy() 自动继承
        if config.max_parents is not None:
            prior_graph.max_parents = config.max_parents

        # 初始化评分模块
        self.mdl_score = MDLScore(data, config.n_states,
                                   penalty_scale=config.mdl_penalty_scale)
        self.sdiff_score = StructuralDiffScore(prior_graph)
        self.cs = CompositeScore(self.mdl_score, self.sdiff_score)

        # 生成权重向量和邻居结构
        H = config.n_weight_vectors - 1  # Das-Dennis 分区数
        self.weights = das_dennis_weights(2, H)
        # 权重向量数量可能因舍入与配置不同，修正种群大小
        self.pop_size = len(self.weights)
        self.neighbors = compute_neighborhood(self.weights, config.n_neighbors)

        # 生成初始种群
        self.population = generate_initial_population(
            prior_graph, self.sdiff_score, config,
            rng=random.Random(config.random_seed),
        )
        # 若种群大小与权重数不同，截断或复制
        if len(self.population) < self.pop_size:
            while len(self.population) < self.pop_size:
                self.population.append(prior_graph.copy())
        self.population = self.population[: self.pop_size]

        # 初始评估
        evals = [self._evaluate(g) for g in self.population]
        self.F = np.array([e[0] for e in evals])
        self.node_scores: list[dict[int, tuple[float, float]]] = [e[1] for e in evals]
        self.ideal = np.min(self.F, axis=0).copy()

    def _evaluate(self, graph: DirectedGraph) -> tuple[np.ndarray, dict[int, tuple[float, float]]]:
        """计算图的 (mdl, sdiff) 目标向量和逐节点评分缓存。

        Returns:
            (f_values, node_scores) 其中 f_values=(mdl, sdiff),
            node_scores={node: (mdl_val, sdiff_val)}
        """
        mdl, sd = self.cs.full_evaluate(graph)
        # 读取 full_evaluate 写入的逐节点缓存
        scores = {}
        for node in range(self.n_nodes):
            scores[node] = (self.cs.get_node_mdl(node), self.cs.get_node_sdiff(node))
        return np.array([mdl, sd]), scores

    def run(self) -> MOEADResult:
        """执行 MOEA/D 主循环。"""
        config = self.config
        rng = random.Random(config.random_seed)
        history: list[np.ndarray] = []
        t0 = time.time()

        for gen in range(config.n_generations):
            # 进度条
            pct = (gen + 1) / config.n_generations
            bar_len = 30
            filled = int(bar_len * pct)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"\r  [{bar}] {gen+1}/{config.n_generations} ({pct:.0%})", end="", flush=True)

            # 更新当前种群的 nadir point
            nadir = compute_nadir(self.F)

            for i in range(self.pop_size):
                # 1. 选择父代
                k, l = select_parents(
                    self.pop_size, i, self.neighbors,
                    config.prob_neighbor_mating, rng,
                )

                # 2. 交叉（使用缓存逐节点评分，避免重复计算）
                child = crossover(
                    self.population[k],
                    self.population[l],
                    self.cs,
                    self.weights[i],
                    self.ideal,
                    nadir,
                    config.eps,
                    rng,
                    parent1_scores=self.node_scores[k],
                    parent2_scores=self.node_scores[l],
                )

                # 3. 变异
                if rng.random() < config.mutation_prob:
                    child = mutate(child, config, rng)

                # 4. 评估
                child_f, child_node_scores = self._evaluate(child)

                # 检查对称差约束
                if child_f[1] > config.max_symmetric_diff:
                    continue

                # 5. 更新 ideal point
                self.ideal = update_ideal(child_f, self.ideal)

                # 6. 更新邻居
                neighbor_list = self.neighbors[i].copy()
                rng.shuffle(neighbor_list)
                n_replaced = 0
                for j in neighbor_list:
                    if n_replaced >= config.max_replacements:
                        break
                    g_child = chebyshev_aggregate(
                        child_f, self.weights[j], self.ideal, nadir, config.eps
                    )
                    g_curr = chebyshev_aggregate(
                        self.F[j], self.weights[j], self.ideal, nadir, config.eps
                    )
                    if g_child <= g_curr:
                        self.population[j] = child.copy()
                        self.F[j] = child_f.copy()
                        self.node_scores[j] = child_node_scores.copy()
                        n_replaced += 1

            # 记录当前 Pareto 前沿（用于收敛历史）
            pareto_mask = non_dominated_sort(self.F)
            history.append(self.F[pareto_mask].copy())

        # 进度条结束后换行
        print()
        runtime = time.time() - t0

        # 提取最终 Pareto 前沿
        pareto_mask = non_dominated_sort(self.F)
        pareto_indices = np.where(pareto_mask)[0]

        return MOEADResult(
            pareto_graphs=[self.population[i] for i in pareto_indices],
            pareto_f=self.F[pareto_indices],
            population=self.population,
            population_f=self.F,
            history=history,
            ideal=self.ideal,
            config=config,
            node_names=self.node_names,
            runtime=runtime,
        )
