from __future__ import annotations

import random

from src.graph import DirectedGraph


class PriorNetwork:
    """先验贝叶斯网络的加载与扰动。"""

    @classmethod
    def from_pgmpy_model(
        cls, model_name: str, max_cycle_length: int = 3
    ) -> tuple[DirectedGraph, list[str], list[int]]:
        """从 pgmpy 内置 bnlearn 网络加载先验网络。

        Args:
            model_name: bnlearn 网络名称，如 'asia', 'alarm', 'sachs' 等
            max_cycle_length: 图的最大允许环长度

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
            len(node_names), edges, max_cycle_length
        )
        return graph, node_names, n_states

    @classmethod
    def from_bif(
        cls, path: str, max_cycle_length: int = 3
    ) -> tuple[DirectedGraph, list[str], list[int]]:
        """从 BIF 文件加载先验网络。

        Args:
            path: BIF 文件路径
            max_cycle_length: 图的最大允许环长度

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
            len(node_names), edges, max_cycle_length
        )
        return graph, node_names, n_states

    @classmethod
    def from_edges(
        cls,
        n_nodes: int,
        edges: list[tuple[int, int]],
        n_states: list[int],
        node_names: list[str] | None = None,
        max_cycle_length: int = 3,
    ) -> tuple[DirectedGraph, list[str], list[int]]:
        """从边列表构造先验网络。

        Args:
            n_nodes: 节点数
            edges: 边列表
            n_states: 每个节点的取值数
            node_names: 节点名称（可选，默认使用索引字符串）
            max_cycle_length: 图的最大允许环长度

        Returns:
            (graph, node_names, n_states)
        """
        if node_names is None:
            node_names = [str(i) for i in range(n_nodes)]
        graph = DirectedGraph.from_edges(n_nodes, edges, max_cycle_length)
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
                    g.reverse_edge(u, v)

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
