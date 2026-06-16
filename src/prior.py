from __future__ import annotations

import math
import random

from src.graph import DirectedGraph


def compute_n_perturb(n_edges: int) -> int:
    """计算先验扰动次数的默认值。

    使用 sqrt(E) 实现子线性缩放，保底 3 次。零边时不扰动。
    """
    if n_edges == 0:
        return 0
    return max(3, int(math.sqrt(n_edges)))


class PriorNetwork:
    """先验贝叶斯网络的加载与扰动。"""

    @classmethod
    def from_pgmpy_model(
        cls, model_name: str,
    ) -> tuple[DirectedGraph, list[str], list[int]]:
        """从 pgmpy 内置 bnlearn 网络加载先验网络。

        Args:
            model_name: bnlearn 网络名称，如 'asia', 'alarm', 'sachs' 等

        Returns:
            (graph, node_names, n_states)
        """
        from pgmpy.example_models import load_model

        # 同时支持 'asia' 和 'bnlearn/asia' 两种格式
        if not model_name.startswith("bnlearn/") and not model_name.startswith("bnrep/"):
            model_name = f"bnlearn/{model_name}"
        model = load_model(model_name)
        node_names = list(model.nodes())
        name_to_idx = {name: i for i, name in enumerate(node_names)}

        # 获取每个节点的取值数
        cardinalities = model.get_cardinality()
        n_states = [int(cardinalities[name]) for name in node_names]

        edges = [
            (name_to_idx[u], name_to_idx[v])
            for u, v in model.edges()
        ]

        graph = DirectedGraph.from_edges(
            len(node_names), edges
        )
        return graph, node_names, n_states

    @classmethod
    def from_bif(
        cls, path: str,
    ) -> tuple[DirectedGraph, list[str], list[int]]:
        """从 BIF 文件加载先验网络。

        Args:
            path: BIF 文件路径

        Returns:
            (graph, node_names, n_states)
        """
        from pgmpy.readwrite import BIFReader

        reader = BIFReader(path)
        model = reader.get_model()
        node_names = list(model.nodes())
        name_to_idx = {name: i for i, name in enumerate(node_names)}

        cardinalities = model.get_cardinality()
        n_states = [int(cardinalities[name]) for name in node_names]

        edges = [
            (name_to_idx[u], name_to_idx[v])
            for u, v in model.edges()
        ]

        graph = DirectedGraph.from_edges(
            len(node_names), edges
        )
        return graph, node_names, n_states

    @classmethod
    def from_edges(
        cls,
        n_nodes: int,
        edges: list[tuple[int, int]],
        n_states: list[int],
        node_names: list[str] | None = None,
    ) -> tuple[DirectedGraph, list[str], list[int]]:
        """从边列表构造先验网络。

        Args:
            n_nodes: 节点数
            edges: 边列表
            n_states: 每个节点的取值数
            node_names: 节点名称（可选，默认使用索引字符串）

        Returns:
            (graph, node_names, n_states)
        """
        if node_names is None:
            node_names = [str(i) for i in range(n_nodes)]
        graph = DirectedGraph.from_edges(n_nodes, edges)
        return graph, node_names, n_states

    @staticmethod
    def perturb(
        graph: DirectedGraph,
        n_changes: int,
        seed: int | None = None,
    ) -> DirectedGraph:
        """对图施加 n_changes 次随机边操作（加边/删边/反转边）。

        用于在给定先验网络的基础上生成变体。

        Args:
            graph: 原始图
            n_changes: 修改次数
            seed: 随机种子

        Returns:
            修改后的新图
        """
        rng = random.Random(seed)
        g = graph.copy()
        n = g.n_nodes
        ops = ["add", "remove", "reverse"]

        for _ in range(n_changes):
            op = rng.choice(ops)
            if op == "add":
                u, v = rng.randrange(n), rng.randrange(n)
                if not g._would_create_cycle(u, v):
                    g.add_edge(u, v)
            elif op == "remove":
                edges = g.get_edges()
                if edges:
                    u, v = rng.choice(edges)
                    g.remove_edge(u, v)
            else:  # reverse
                edges = g.get_edges()
                if edges:
                    u, v = rng.choice(edges)
                    g.remove_edge(u, v)
                    if not g._would_create_cycle(v, u):
                        g.add_edge(v, u)
                    else:
                        g.adj[u, v] = 1  # 回滚，直接恢复无需检测

        return g

    @staticmethod
    def construct_perturbed(
        gt_graph: DirectedGraph,
        delete_frac: float,
        seed: int | None = None,
        max_parents: int | None = None,
    ) -> tuple[DirectedGraph, dict]:
        """按目标 SHD 构造扰动先验。

        从 GT 中删除 d = round(delete_frac × E) 条边，
        再添加 d 条不在 GT 中的新边（保 DAG + max_parents）。
        SHD(result, GT) ≈ 2d。

        Returns:
            (graph, {"edges_deleted": int, "edges_added": int, "shd_from_gt": int})
        """
        rng = random.Random(seed)
        g = gt_graph.copy()
        if max_parents is not None:
            g.max_parents = max_parents

        gt_edges = set(gt_graph.get_edges())
        n = g.n_nodes
        d = round(delete_frac * len(gt_edges))

        # 1. 随机删除 d 条边
        to_delete = rng.sample(sorted(gt_edges), d)
        for u, v in to_delete:
            g.remove_edge(u, v)

        # 2. 随机添加 d 条新边（不在 GT 边集中）
        added = 0
        for _ in range(d * 20):
            if added >= d:
                break
            u = rng.randrange(n)
            v = rng.randrange(n)
            if u == v or (u, v) in gt_edges:
                continue
            if g.add_edge(u, v):  # 内部检查: 已存在/环/max_parents
                added += 1

        current_edges = set(g.get_edges())
        return g, {
            "edges_deleted": d,
            "edges_added": added,
            "shd_from_gt": len(current_edges ^ gt_edges),
        }

    @staticmethod
    def random_dag(
        n_nodes: int,
        density: float,
        seed: int | None = None,
        max_parents: int | None = None,
    ) -> DirectedGraph:
        """生成与指定密度匹配的随机 DAG。

        随机拓扑序 + Bernoulli(density) 加边，天然无环。
        期望边数 ≈ density × C(n,2)。

        Args:
            n_nodes: 节点数
            density: 边密度 p = E_target / C(n,2)
            seed: 随机种子
            max_parents: 每个节点最大父节点数

        Returns:
            随机 DAG
        """
        rng = random.Random(seed)
        g = DirectedGraph(n_nodes, max_parents=max_parents)

        # 随机拓扑序
        order = list(range(n_nodes))
        rng.shuffle(order)

        # 边只从 order 前方指向后方 → 天然 DAG，无需判环
        parent_count = [0] * n_nodes
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                v = order[j]
                if max_parents is not None and parent_count[v] >= max_parents:
                    continue
                if rng.random() < density:
                    g.adj[order[i], v] = 1
                    parent_count[v] += 1

        return g

    @staticmethod
    def generate_data(
        model_name_or_graph,
        n_samples: int,
        node_names: list[str] | None = None,
        n_states: list[int] | None = None,
        seed: int | None = None,
    ) -> tuple["np.ndarray", list[str], list[int]]:
        """从贝叶斯网络生成模拟数据。

        Args:
            model_name_or_graph: bnlearn 网络名或 pgmpy BayesianNetwork
            n_samples: 样本数
            node_names: 节点名（仅在传入 graph 时需要）
            n_states: 各节点取值数（仅在传入 graph 时需要）
            seed: 随机种子

        Returns:
            (data, node_names, n_states)
        """
        import numpy as np
        from pgmpy.example_models import load_model
        from pgmpy.sampling import BayesianModelSampling

        if isinstance(model_name_or_graph, str):
            name = model_name_or_graph
            if not name.startswith("bnlearn/") and not name.startswith("bnrep/"):
                name = f"bnlearn/{name}"
            model = load_model(name)
        else:
            model = model_name_or_graph

        node_names = list(model.nodes())
        cardinalities = model.get_cardinality()
        n_states = [int(cardinalities[name]) for name in node_names]

        sampler = BayesianModelSampling(model)
        df = sampler.forward_sample(size=n_samples, seed=seed)
        # 将类别数据转为整数编码
        data = np.zeros((n_samples, len(node_names)), dtype=np.int32)
        for i, name in enumerate(node_names):
            # 获取所有可能的类别
            states = model.states[name]
            mapping = {s: j for j, s in enumerate(states)}
            data[:, i] = df[name].map(mapping).values

        return data, node_names, n_states
