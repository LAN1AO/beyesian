from __future__ import annotations


class CompositeScore:
    """增量评分的组合评分器。

    维护每个节点的 MDL 和结构对称差缓存，仅重算被标记为 dirty 的节点。
    利用评分的可分解性提高效率:
    - 修改边 (p→i) 时，仅节点 i 的父集发生变化
    - 反转边 (u→v) 时，节点 v 和 u 的父集都发生变化
    """

    def __init__(self, mdl_score, sdiff_score):
        self._mdl = mdl_score
        self._sdiff = sdiff_score
        self.n_nodes = mdl_score.n_nodes

        # 每节点缓存
        self._mdl_cache: dict[int, float] = {}
        self._sdiff_cache: dict[int, float] = {}
        self._dirty: set[int] = set(range(self.n_nodes))

        # 全图缓存值
        self._total_mdl: float = 0.0
        self._total_sdiff: float = 0.0

    # ── 标记脏节点 ──────────────────────────────────────────

    def mark_dirty(self, nodes: list[int]) -> None:
        """标记指定节点为脏（需重算）。"""
        for n in nodes:
            self._dirty.add(n)

    def mark_all_dirty(self) -> None:
        """标记所有节点为脏。"""
        self._dirty.update(range(self.n_nodes))

    # ── 评分获取 ─────────────────────────────────────────────

    def scores_vector(self, graph) -> tuple[float, float]:
        """返回 (total_mdl, total_sdiff)，仅重算脏节点。"""
        if self._dirty:
            dirty_list = list(self._dirty)
            new_mdl = self._mdl.score_nodes(graph, dirty_list)
            new_sdiff = self._sdiff.score_nodes(graph, dirty_list)

            for node in dirty_list:
                old_mdl = self._mdl_cache.get(node, 0.0)
                old_sdiff = self._sdiff_cache.get(node, 0.0)
                self._total_mdl += new_mdl[node] - old_mdl
                self._total_sdiff += new_sdiff[node] - old_sdiff
                self._mdl_cache[node] = new_mdl[node]
                self._sdiff_cache[node] = new_sdiff[node]

            self._dirty.clear()

        return self._total_mdl, self._total_sdiff

    def node_pair_score(
        self, graph, node: int, parent_set: list[int]
    ) -> tuple[float, float]:
        """计算单个节点在给定父集下的 (mdl, sdiff)，不影响缓存。"""
        mdl = self._mdl.score_node(node, parent_set)
        sdiff = self._sdiff.score_node(node, parent_set)
        return mdl, sdiff

    def full_evaluate(self, graph) -> tuple[float, float]:
        """全量重算评分（忽略缓存）。"""
        for node in range(self.n_nodes):
            parents = graph.get_parents(node)
            self._mdl_cache[node] = self._mdl.score_node(node, parents)
            self._sdiff_cache[node] = self._sdiff.score_node(node, parents)
        self._total_mdl = sum(self._mdl_cache.values())
        self._total_sdiff = sum(self._sdiff_cache.values())
        self._dirty.clear()
        return self._total_mdl, self._total_sdiff

    # ── 查询 ─────────────────────────────────────────────────

    def get_cached_mdl(self) -> float:
        return self._total_mdl

    def get_cached_sdiff(self) -> float:
        return self._total_sdiff

    def get_node_mdl(self, node: int) -> float:
        return self._mdl_cache.get(node, 0.0)

    def get_node_sdiff(self, node: int) -> float:
        return self._sdiff_cache.get(node, 0.0)

    @property
    def mdl(self):
        return self._mdl

    @property
    def sdiff(self):
        return self._sdiff
