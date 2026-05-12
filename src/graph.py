from __future__ import annotations

import numpy as np


class DirectedGraph:
    """带有放松无环约束的有向图。

    允许长度 ≤ max_cycle_length 的环，禁止更长的环。
    使用 numpy 邻接矩阵 (n×n, int8) 存储图结构。
    """

    def __init__(self, n_nodes: int, max_cycle_length: int = 3):
        if max_cycle_length not in (2, 3, 4):
            raise ValueError(f"max_cycle_length 必须为 2, 3, 或 4, 当前值: {max_cycle_length}")
        self.n_nodes = n_nodes
        self.max_cycle_length = max_cycle_length
        self.adj = np.zeros((n_nodes, n_nodes), dtype=np.int8)

    # ── 工厂方法 ──────────────────────────────────────────────

    @classmethod
    def from_edges(
        cls, n_nodes: int, edges: list[tuple[int, int]], max_cycle_length: int = 3
    ) -> DirectedGraph:
        g = cls(n_nodes, max_cycle_length)
        for u, v in edges:
            if not g.add_edge(u, v):
                raise ValueError(
                    f"边 {u}→{v} 会创建长度超过 {max_cycle_length} 的环"
                )
        return g

    @classmethod
    def from_adj_matrix(
        cls, adj: np.ndarray, max_cycle_length: int = 3
    ) -> DirectedGraph:
        n = adj.shape[0]
        g = cls(n, max_cycle_length)
        g.adj = adj.astype(np.int8).copy()
        if not g._is_valid():
            raise ValueError("邻接矩阵存在超过允许长度的环")
        return g

    def copy(self) -> DirectedGraph:
        g = DirectedGraph(self.n_nodes, self.max_cycle_length)
        g.adj = self.adj.copy()
        return g

    # ── 边操作 (全部带环检测) ────────────────────────────────

    def add_edge(self, u: int, v: int) -> bool:
        """尝试添加边 u→v，若产生非法环则返回 False。"""
        if u == v:
            return False  # 禁止自环
        if self.adj[u, v]:
            return True  # 边已存在
        cycle_len = self._would_create_cycle(u, v)
        if cycle_len is not None and cycle_len > self.max_cycle_length:
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

    def _would_create_cycle(self, u: int, v: int) -> int | None:
        """检查添加 u→v 是否会形成环以及环的长度。

        从 v 出发做限界 BFS，检查能否到达 u。
        若能到达 u，返回新环的长度 (路径长度 + 1)；
        若不能或路径过长，返回 None。

        复杂度: O(b^d)，b 为平均出度，d = max_cycle_length ≤ 4。
        """
        visited = {v}
        # queue: (node, depth_from_v)
        queue = [(v, 0)]
        for node, depth in queue:
            if depth >= self.max_cycle_length:
                continue
            for child in self.get_children(node):
                if child == u:
                    return depth + 2  # path v→...→u + edge u→v
                if child not in visited:
                    visited.add(child)
                    queue.append((child, depth + 1))
        return None

    def has_cycle_exceeding(self, max_len: int) -> bool:
        """检查图中是否存在长度超过 max_len 的环 (用于校验)。"""
        for start in range(self.n_nodes):
            # DFS 检测从 start 出发回到 start 的路径
            stack = [(start, [start])]
            while stack:
                node, path = stack.pop()
                if len(path) > max_len + 1:
                    return True
                for child in self.get_children(node):
                    if child == start and len(path) > max_len:
                        return True
                    if child not in path:
                        stack.append((child, path + [child]))
        return False

    def _is_valid(self) -> bool:
        """校验图是否满足环长度约束。"""
        return not self.has_cycle_exceeding(self.max_cycle_length)

    def has_cycle_with_length(self, length: int) -> bool:
        """检查是否存在精确长度为 length 的环。"""
        for start in range(self.n_nodes):
            stack = [(start, [start])]
            while stack:
                node, path = stack.pop()
                if len(path) > length:
                    continue
                for child in self.get_children(node):
                    if child == start and len(path) == length:
                        return True
                    if child not in path:
                        stack.append((child, path + [child]))
        return False

    # ── 工具方法 ──────────────────────────────────────────────

    def to_pgmpy_edges(self) -> list[tuple[str, str]]:
        """转为 pgmpy 格式的边列表 (需要外部提供节点名称映射)。"""
        edges = self.get_edges()
        return [(str(u), str(v)) for u, v in edges]

    def __repr__(self) -> str:
        n_edges = int(np.sum(self.adj))
        return (
            f"DirectedGraph(n={self.n_nodes}, edges={n_edges}, "
            f"max_cycle={self.max_cycle_length})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DirectedGraph):
            return False
        return (
            self.n_nodes == other.n_nodes
            and self.max_cycle_length == other.max_cycle_length
            and np.array_equal(self.adj, other.adj)
        )
