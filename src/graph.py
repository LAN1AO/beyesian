from __future__ import annotations

import numpy as np


class DirectedGraph:
    """有向无环图 (DAG)。

    所有边操作均带环检测，保证图始终无环。
    可选限制每个节点的最大父节点数 max_parents。
    使用 numpy 邻接矩阵 (n×n, int8) 存储图结构。
    """

    def __init__(self, n_nodes: int, max_parents: int | None = None):
        self.n_nodes = n_nodes
        self.max_parents = max_parents
        self.adj = np.zeros((n_nodes, n_nodes), dtype=np.int8)

    # ── 工厂方法 ──────────────────────────────────────────────

    @classmethod
    def from_edges(
        cls, n_nodes: int, edges: list[tuple[int, int]],
        max_parents: int | None = None,
    ) -> DirectedGraph:
        g = cls(n_nodes, max_parents=max_parents)
        for u, v in edges:
            if not g.add_edge(u, v):
                raise ValueError(f"边 {u}→{v} 会产生环")
        return g

    @classmethod
    def from_adj_matrix(
        cls, adj: np.ndarray, max_parents: int | None = None,
    ) -> DirectedGraph:
        n = adj.shape[0]
        g = cls(n, max_parents=max_parents)
        g.adj = adj.astype(np.int8).copy()
        if g.count_cycles() > 0:
            raise ValueError("邻接矩阵存在环")
        return g

    def copy(self) -> DirectedGraph:
        g = DirectedGraph(self.n_nodes, max_parents=self.max_parents)
        g.adj = self.adj.copy()
        return g

    # ── 边操作 (全部带环检测) ────────────────────────────────

    def add_edge(self, u: int, v: int) -> bool:
        """尝试添加边 u→v，若产生环或超过最大父节点数则返回 False。"""
        if u == v:
            return False  # 禁止自环
        if self.adj[u, v]:
            return True  # 边已存在
        if self.max_parents is not None and self.get_in_degree(v) >= self.max_parents:
            return False
        if self._would_create_cycle(u, v):
            return False
        self.adj[u, v] = 1
        return True

    def remove_edge(self, u: int, v: int) -> bool:
        """删除边 u→v，若边存在返回 True。"""
        existed = bool(self.adj[u, v])
        self.adj[u, v] = 0
        return existed

    def reverse_edge(self, u: int, v: int) -> bool:
        """反转边 u→v 为 v→u，若非法则回滚并返回 False。"""
        if not self.adj[u, v]:
            return False
        self.adj[u, v] = 0
        if self.add_edge(v, u):
            return True
        # 回滚: 恢复原边
        self.adj[u, v] = 1
        return False

    def has_edge(self, u: int, v: int) -> bool:
        return bool(self.adj[u, v])

    # ── 邻接查询 ──────────────────────────────────────────────

    def get_parents(self, node: int) -> list[int]:
        """返回 node 的所有父节点索引。"""
        return np.where(self.adj[:, node] == 1)[0].tolist()

    def get_children(self, node: int) -> list[int]:
        """返回 node 的所有子节点索引。"""
        return np.where(self.adj[node, :] == 1)[0].tolist()

    def get_edges(self) -> list[tuple[int, int]]:
        """返回所有边 (u, v) 的列表。"""
        rows, cols = np.where(self.adj == 1)
        return list(zip(rows.tolist(), cols.tolist()))

    def get_in_degree(self, node: int) -> int:
        return int(np.sum(self.adj[:, node]))

    def get_out_degree(self, node: int) -> int:
        return int(np.sum(self.adj[node, :]))

    # ── 环检测 ────────────────────────────────────────────────

    def _would_create_cycle(self, u: int, v: int) -> bool:
        """检查添加 u→v 是否会形成环（DFS 从 v 出发检查能否到达 u）。"""
        visited = {v}
        stack = [v]
        while stack:
            node = stack.pop()
            for child in self.get_children(node):
                if child == u:
                    return True
                if child not in visited:
                    visited.add(child)
                    stack.append(child)
        return False

    def _iter_cycles(self):
        """迭代图中所有简单环，每个环只产出一次（以环中最小节点为起点）。

        Yields: (cycle_length, path_list)
        """
        for start in range(self.n_nodes):
            stack = [(start, [start])]
            while stack:
                node, path = stack.pop()
                for child in self.get_children(node):
                    if child == start:
                        if min(path) == start:  # 只在最小节点处产出，避免重复
                            yield len(path), path
                    elif child not in path:
                        stack.append((child, path + [child]))

    def count_cycles(self) -> int:
        return sum(1 for _ in self._iter_cycles())

    # ── 工具方法 ──────────────────────────────────────────────

    def to_pgmpy_edges(self) -> list[tuple[str, str]]:
        """转为 pgmpy 格式的边列表 (需要外部提供节点名称映射)。"""
        edges = self.get_edges()
        return [(str(u), str(v)) for u, v in edges]

    def __repr__(self) -> str:
        n_edges = int(np.sum(self.adj))
        mp = f", max_parents={self.max_parents}" if self.max_parents else ""
        return f"DirectedGraph(n={self.n_nodes}, edges={n_edges}{mp})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DirectedGraph):
            return False
        return (
            self.n_nodes == other.n_nodes
            and self.max_parents == other.max_parents
            and np.array_equal(self.adj, other.adj)
        )
