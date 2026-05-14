from __future__ import annotations

import numpy as np


class DirectedGraph:
    """带有放松无环约束的有向图。

    禁止长度 ≤ max_forbidden_cycle 的短环（易检测），允许更长的环。
    例如 n=3 时: 禁止 2-环和 3-环，允许 4+ 环（留待后续 DAG 转化处理）。
    可选限制每个节点的最大父节点数 max_parents。
    使用 numpy 邻接矩阵 (n×n, int8) 存储图结构。
    """

    def __init__(self, n_nodes: int, max_forbidden_cycle: int = 3,
                 max_parents: int | None = None):
        if max_forbidden_cycle not in (2, 3, 4):
            raise ValueError(
                f"max_forbidden_cycle 必须为 2, 3, 或 4, "
                f"当前值: {max_forbidden_cycle}"
            )
        self.n_nodes = n_nodes
        self.max_forbidden_cycle = max_forbidden_cycle
        self.max_parents = max_parents
        self.adj = np.zeros((n_nodes, n_nodes), dtype=np.int8)

    # ── 工厂方法 ──────────────────────────────────────────────

    @classmethod
    def from_edges(
        cls, n_nodes: int, edges: list[tuple[int, int]], max_forbidden_cycle: int = 3,
        max_parents: int | None = None,
    ) -> DirectedGraph:
        g = cls(n_nodes, max_forbidden_cycle, max_parents=max_parents)
        for u, v in edges:
            if not g.add_edge(u, v):
                raise ValueError(
                    f"边 {u}→{v} 会创建长度 ≤ {max_forbidden_cycle} 的被禁环"
                )
        return g

    @classmethod
    def from_adj_matrix(
        cls, adj: np.ndarray, max_forbidden_cycle: int = 3,
        max_parents: int | None = None,
    ) -> DirectedGraph:
        n = adj.shape[0]
        g = cls(n, max_forbidden_cycle, max_parents=max_parents)
        g.adj = adj.astype(np.int8).copy()
        if not g._is_valid():
            raise ValueError("邻接矩阵存在被禁的短环")
        return g

    def copy(self) -> DirectedGraph:
        g = DirectedGraph(self.n_nodes, self.max_forbidden_cycle,
                          max_parents=self.max_parents)
        g.adj = self.adj.copy()
        return g

    # ── 边操作 (全部带环检测) ────────────────────────────────

    def add_edge(self, u: int, v: int) -> bool:
        """尝试添加边 u→v，若产生被禁短环或超过最大父节点数则返回 False。"""
        if u == v:
            return False  # 禁止自环
        if self.adj[u, v]:
            return True  # 边已存在
        if self.max_parents is not None and self.get_in_degree(v) >= self.max_parents:
            return False
        cycle_len = self._would_create_cycle(u, v)
        if cycle_len is not None and cycle_len <= self.max_forbidden_cycle:
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

        从 v 出发做限界 BFS，仅检测长度 ≤ max_forbidden_cycle 的环
        （更长的环被允许，无需检测）。

        复杂度: O(b^d)，b 为平均出度，d ≤ 3。
        """
        # BFS 最大深度: 深度 d 可检测到 d+2 长度的环
        # 只需检测 ≤ max_forbidden_cycle 的环 → 最大深度 = max_forbidden_cycle - 2
        max_depth = self.max_forbidden_cycle - 2
        if max_depth < 0:
            return None  # 若 max_forbidden_cycle=2，仅需检测 2-环，BFS 深度 0 即可

        visited = {v}
        queue = [(v, 0)]
        for node, depth in queue:
            if depth > max_depth:
                continue
            for child in self.get_children(node):
                if child == u:
                    return depth + 2  # 环长度 = path v→...→u + edge u→v
                if child not in visited:
                    visited.add(child)
                    queue.append((child, depth + 1))
        return None

    def has_forbidden_cycle(self, max_forbidden: int) -> bool:
        """检查图中是否存在被禁短环（长度 ≤ max_forbidden），用于校验。"""
        for start in range(self.n_nodes):
            stack = [(start, [start])]
            while stack:
                node, path = stack.pop()
                if len(path) > max_forbidden:
                    continue
                for child in self.get_children(node):
                    if child == start and 2 <= len(path) <= max_forbidden:
                        return True
                    if child not in path:
                        stack.append((child, path + [child]))
        return False

    def _is_valid(self) -> bool:
        """校验图是否满足环长度约束（无被禁短环）。"""
        return not self.has_forbidden_cycle(self.max_forbidden_cycle)

    def count_cycles(self) -> int:
        """统计图中所有环的数量（忽略长度限制，计数所有简单环）。"""
        count = 0
        for start in range(self.n_nodes):
            stack = [(start, [start])]
            while stack:
                node, path = stack.pop()
                for child in self.get_children(node):
                    if child == start:
                        count += 1
                    elif child not in path:
                        stack.append((child, path + [child]))
        return count

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
        mp = f", max_parents={self.max_parents}" if self.max_parents else ""
        return (
            f"DirectedGraph(n={self.n_nodes}, edges={n_edges}, "
            f"forbid_cycle≤{self.max_forbidden_cycle}{mp})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DirectedGraph):
            return False
        return (
            self.n_nodes == other.n_nodes
            and self.max_forbidden_cycle == other.max_forbidden_cycle
            and self.max_parents == other.max_parents
            and np.array_equal(self.adj, other.adj)
        )
