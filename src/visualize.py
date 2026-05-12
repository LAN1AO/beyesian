from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use("Agg")  # 非交互后端，避免 GUI 依赖


def plot_pareto_front(
    result,
    title: str = "Pareto Front",
    save_path: str | None = None,
    show_prior: bool = True,
) -> plt.Figure:
    """绘制 Pareto 前沿散点图。

    Args:
        result: MOEADResult 实例
        title: 图表标题
        save_path: 若给定，保存图表到此路径
        show_prior: 是否标注先验网络位置

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # 绘制 Pareto 前沿
    f = result.pareto_f
    ax.scatter(
        f[:, 0], f[:, 1],
        c="steelblue", s=40, alpha=0.8,
        edgecolors="navy", linewidth=0.5,
        label="Pareto Front",
    )

    # 标注先验网络位置
    if show_prior and result.config.data is not None:
        # 先验网络的对称差固定为 0
        prior_sdiff = 0.0
        # 我们需要计算先验网络的 MDL
        # 使用 result.population[0] 中的 score 重新计算
        from src.score import MDLScore, StructuralDiffScore
        from src.graph import DirectedGraph

        # 从结果中获取先验 MDL: 找 Sdiff=0 的 Pareto 解的 MDL
        zero_sdiff = f[f[:, 1] == 0]
        if len(zero_sdiff) > 0:
            prior_mdl = zero_sdiff[0, 0]
            ax.scatter(
                [prior_mdl], [0],
                c="red", s=100, marker="*",
                edgecolors="darkred", linewidth=1,
                label="Prior Network",
                zorder=5,
            )

    ax.set_xlabel("MDL Score", fontsize=12)
    ax.set_ylabel("Structural Symmetric Difference", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Pareto front chart saved to: {save_path}")

    return fig


def plot_convergence(
    result,
    title: str = "Convergence History",
    save_path: str | None = None,
) -> plt.Figure:
    """绘制收敛曲线：Pareto 前沿随世代数的演变。

    使用 Hypervolume 代理指标（非支配解覆盖的面积）。

    Args:
        result: MOEADResult 实例
        title: 图表标题
        save_path: 若给定，保存图表到此路径

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    history = result.history
    generations = list(range(len(history)))

    # 每个世代的 Pareto 解数量
    pareto_sizes = [len(h) for h in history]
    ax.plot(generations, pareto_sizes, "b-", linewidth=1.5, label="Pareto Size")

    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Number of Pareto Solutions", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Convergence chart saved to: {save_path}")

    return fig


def plot_objective_convergence(
    result,
    title: str = "Objective Convergence",
    save_path: str | None = None,
) -> plt.Figure:
    """绘制各目标的最优值随世代收敛曲线。

    跟踪 ideal point 的两个分量：最优 MDL 和最优对称差。

    Args:
        result: MOEADResult 实例
        title: 图表标题
        save_path: 若给定，保存图表到此路径

    Returns:
        matplotlib Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    history = result.history

    # 每代最优 MDL (越小越好)
    best_mdl = [np.min(h[:, 0]) if len(h) > 0 else np.nan for h in history]
    ax1.plot(best_mdl, "g-", linewidth=1.5)
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Best MDL")
    ax1.set_title("MDL Convergence")
    ax1.grid(True, alpha=0.3)

    # 每代最优 Sdiff (越小越好)
    best_sdiff = [np.min(h[:, 1]) if len(h) > 0 else np.nan for h in history]
    ax2.plot(best_sdiff, "r-", linewidth=1.5)
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Best Sdiff")
    ax2.set_title("Structural Diff Convergence")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Objective convergence chart saved to: {save_path}")

    return fig


def plot_network(
    graph,
    node_names: list[str],
    title: str = "Network Structure",
    save_path: str | None = None,
) -> plt.Figure:
    """绘制贝叶斯网络结构图。"""
    import networkx as nx

    fig, ax = plt.subplots(figsize=(10, 8))

    G = nx.DiGraph()
    G.add_nodes_from(range(graph.n_nodes))
    for u, v in graph.get_edges():
        G.add_edge(u, v)

    labels = {i: name for i, name in enumerate(node_names)}
    pos = nx.spring_layout(G, seed=42, k=2)

    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=800, ax=ax)
    nx.draw_networkx_labels(G, pos, labels, font_size=9, ax=ax)
    nx.draw_networkx_edges(
        G, pos, arrowstyle="->", arrowsize=15,
        edge_color="gray", alpha=0.7, ax=ax,
    )

    ax.set_title(title, fontsize=14)
    ax.axis("off")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Network structure saved to: {save_path}")

    return fig
