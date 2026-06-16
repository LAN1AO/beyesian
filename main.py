"""多目标贝叶斯网络结构学习 — CLI 入口。

使用 MOEA/D + 切比雪夫分解，同时优化 MDL 评分和结构对称差。
先验网络和数据需通过 scripts/prepare_data.py 预生成。

单次运行:
  python3 main.py --prior-file data/priors/alarm_empty.pkl \\
      --data-file data/synthetic/alarm_N5000.npy \\
      --ground-truth data/ground_truth/alarm_graph.pkl \\
      --pop-size 100 --generations 500

并行 batch 运行 (N=20):
  python3 main.py --prior-file data/priors/alarm_empty.pkl \\
      --data-file data/synthetic/alarm_N5000.npy \\
      --batch 20 --workers 8 --pop-size 100 --generations 500
"""

from __future__ import annotations

import argparse
import os
import pickle
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from src.config import MOEADConfig
from src.metrics import compute_f1, compute_f1_skeleton, compute_shd, compute_shd_skeleton
from src.moead import MOEAD
from src.visualize import (
    plot_convergence,
    plot_objective_convergence,
    plot_pareto_front,
    plot_network,
)


def main():
    parser = argparse.ArgumentParser(
        description="MOEA/D 多目标贝叶斯网络结构学习"
    )

    # 先验网络与数据 (必填)
    parser.add_argument(
        "--prior-file", type=str, required=True,
        help="预生成先验网络文件 (.pkl)",
    )
    parser.add_argument(
        "--data-file", type=str, required=True,
        help="预生成数据文件 (.npy)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="网络名称标签 (默认: 从数据文件推断)",
    )
    parser.add_argument(
        "--ground-truth", type=str, default=None,
        help="真实图文件 (.pkl)，用于标注 GT 位置并计算 SHD/F1 指标",
    )

    # MOEA/D 参数
    parser.add_argument(
        "--max-parents", type=int, default=None,
        help="每个节点最大父节点数 (默认: 不限制)",
    )
    parser.add_argument(
        "--mdl-penalty", type=float, default=1.0,
        help="MDL 惩罚项缩放因子 (默认 1.0=标准BIC, 越小惩罚越轻)",
    )
    parser.add_argument(
        "--max-sdiff", type=int, default=None,
        help="结构对称差上限 (默认: 自动计算 = n×max_parents+E_prior)",
    )
    parser.add_argument(
        "--pop-size", type=int, default=50,
        help="种群大小 / 权重向量数 (默认: 50)",
    )
    parser.add_argument(
        "--neighbors", type=int, default=10,
        help="邻居数量 T (默认: 10)",
    )
    parser.add_argument(
        "--generations", type=int, default=200,
        help="最大世代数 (默认: 200)",
    )
    parser.add_argument(
        "--prob-neighbor", type=float, default=0.9,
        help="从邻居选父代的概率 delta (默认: 0.9)",
    )
    parser.add_argument(
        "--max-replace", type=int, default=2,
        help="每个子代最多替换邻居数 nr (默认: 2)",
    )
    parser.add_argument(
        "--mutation-prob", type=float, default=0.3,
        help="变异概率 (默认: 0.3)",
    )
    parser.add_argument(
        "--mutation-ops-min", type=int, default=2,
        help="每次变异最少边操作次数 (默认: 2)",
    )
    parser.add_argument(
        "--mutation-ops-max", type=int, default=6,
        help="每次变异最多边操作次数 (默认: 6)",
    )
    parser.add_argument(
        "--crossover-type", type=str, default="sequential",
        choices=["sequential", "no-cycle-check"],
        help="交叉算子类型: sequential (默认) | no-cycle-check",
    )

    # 输出
    parser.add_argument(
        "--output", type=str, default="./output",
        help="输出目录 (默认: ./output)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子 (默认: 42)",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="跳过可视化",
    )
    parser.add_argument(
        "--no-params", action="store_true",
        help="跳过输出 params.json (batch 子进程使用)",
    )
    parser.add_argument(
        "--plot-networks", action="store_true",
        help="输出三个代表解的网络结构图 (默认关闭)",
    )

    # Batch 模式
    parser.add_argument(
        "--batch", type=int, default=None,
        help="并行运行次数（启用 batch 模式）",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="并行 worker 数 (默认: CPU 核心数)",
    )

    args = parser.parse_args()

    if args.batch is not None:
        _run_batch(args)
    else:
        _run_single(args)


# ═══════════════════════════════════════════════════════════════
# 单次运行
# ═══════════════════════════════════════════════════════════════

def _load_prior_file(path: str) -> tuple["DirectedGraph", list[str], list[int], list[int] | None]:
    """加载先验网络文件，兼容旧格式 (tuple) 和新格式 (dict)。

    Returns:
        (graph, node_names, n_states, known_node_indices)
        known_node_indices: None=全部已知(旧格式), list=已知节点索引
    """
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        known = obj.get("known_node_indices", None)
        return obj["graph"], obj["node_names"], obj["n_states"], known
    return obj[0], obj[1], obj[2], None  # 旧格式: 全部节点已知


def _run_single(args):
    """单次 MOEA/D 运行。"""
    os.makedirs(args.output, exist_ok=True)

    # ── 1. 加载先验网络 ──────────────────────────────────────
    print(f"[1/5] 加载先验网络...")
    prior_graph, node_names, n_states, known_node_indices = _load_prior_file(args.prior_file)
    if known_node_indices is not None:
        print(f"  先验: {args.prior_file}"
              f" (已知节点: {len(known_node_indices)}/{len(node_names)})")
    else:
        print(f"  先验: {args.prior_file}")
    print(f"  节点: {len(node_names)}")

    # ── 2. 准备数据 ──────────────────────────────────────────
    print(f"[2/5] 准备数据...")
    data = np.load(args.data_file).astype(np.int32)
    print(f"  数据: {data.shape}")

    # ── 3. 配置 MOEA/D ───────────────────────────────────────
    print(f"[3/5] 配置 MOEA/D...")

    # 自动计算 max_sdiff 理论最大值
    if args.max_sdiff is None:
        n_prior_edges = len(prior_graph.get_edges())
        max_sdiff = _compute_max_sdiff(len(node_names), args.max_parents,
                                        n_prior_edges, known_node_indices)
        n_known = len(known_node_indices) if known_node_indices is not None else len(node_names)
        print(f"  max_sdiff 自动计算: {n_known}×{args.max_parents or (len(node_names)-1)}"
              f"+{n_prior_edges} = {max_sdiff}")
    else:
        max_sdiff = args.max_sdiff

    config = MOEADConfig(
        n_nodes=len(node_names),
        n_states=n_states,
        max_parents=args.max_parents,
        mdl_penalty_scale=args.mdl_penalty,
        max_symmetric_diff=max_sdiff,
        known_node_indices=known_node_indices,
        n_weight_vectors=args.pop_size,
        n_neighbors=args.neighbors,
        n_generations=args.generations,
        prob_neighbor_mating=args.prob_neighbor,
        max_replacements=args.max_replace,
        crossover_type=args.crossover_type,
        mutation_prob=args.mutation_prob,
        mutation_ops_min=args.mutation_ops_min,
        mutation_ops_max=args.mutation_ops_max,
        data=data,
        output_dir=args.output,
        random_seed=args.seed,
    )
    print(f"  种群: {args.pop_size}, 世代: {args.generations}, "
          f"邻居: {args.neighbors}")

    # ── 4. 运行 MOEA/D ───────────────────────────────────────
    print(f"[4/5] 运行 MOEA/D...")
    moead = MOEAD(config, prior_graph, data, node_names)
    result = moead.run()

    print(f"  完成! 耗时: {result.runtime:.1f}s")
    print(f"  Pareto 前沿解数: {len(result.pareto_graphs)}")
    print(f"  Ideal point: MDL={result.ideal[0]:.2f}, Sdiff={result.ideal[1]:.0f}")

    # 打印 Pareto 前沿（合并重复解）
    from collections import Counter
    pf_keys = [(int(np.sum(g.adj)), round(f[0], 2), int(f[1]))
               for g, f in zip(result.pareto_graphs, result.pareto_f)]
    pf_counts = Counter(pf_keys)
    print(f"\n  Pareto 前沿 (唯一解 {len(pf_counts)} 个):")
    print(f"  {'#':>3}  {'Edges':>5}  {'MDL':>12}  {'Sdiff':>5}  {'Count':>5}")
    print(f"  {'-' * 42}")
    for i, ((edges, mdl, sdiff), count) in enumerate(
        sorted(pf_counts.items(), key=lambda x: (x[0][2], x[0][1]))
    ):
        print(f"  {i:>3}  {edges:>5}  {mdl:>12.2f}  {sdiff:>5}  {count:>5}")

    # ── 5. 保存结果和可视化 ──────────────────────────────────
    print(f"\n[5/5] 保存结果...")

    # 保存结果对象
    result_path = os.path.join(args.output, "result.pkl")
    with open(result_path, "wb") as f:
        pickle.dump(result, f)
    print(f"  结果: {result_path}")

    # 加载真实图 (可选)
    gt_graph = None
    gt_pos = None
    gt_edges_set = None
    if args.ground_truth:
        gt_graph, _, _ = _load_prior_file(args.ground_truth)[:3]
        gt_edges = gt_graph.get_edges()
        gt_edges_set = set(gt_edges)
        # 计算 GT 在目标空间中的位置
        from src.score import MDLScore, StructuralDiffScore
        mdl_scorer = MDLScore(data, n_states, penalty_scale=args.mdl_penalty)
        sdiff_scorer = StructuralDiffScore(prior_graph)
        gt_pos = (mdl_scorer.score_graph(gt_graph),
                  sdiff_scorer.score_graph(gt_graph))
        print(f"  真实图: {len(gt_edges)} 边, MDL={gt_pos[0]:.2f}, Sdiff={gt_pos[1]:.0f}")

    # 保存 Pareto 前沿 CSV
    pareto_path = os.path.join(args.output, "pareto_front.csv")
    sorted_pf = sorted(pf_counts.items(), key=lambda x: (x[0][2], x[0][1]))
    if gt_graph is not None:
        header = "index,edges,mdl,sdiff,shd,f1,shd_skel,f1_skel,count\n"
    else:
        header = "index,edges,mdl,sdiff,count\n"
    with open(pareto_path, "w") as f:
        f.write(header)
        for i, ((edges, mdl, sdiff), count) in enumerate(sorted_pf):
            if gt_graph is not None:
                # 找到该唯一解对应的一个实例来计算 SHD/F1
                for g_obj, f_obj in zip(result.pareto_graphs, result.pareto_f):
                    key = (int(np.sum(g_obj.adj)), round(f_obj[0], 2), int(f_obj[1]))
                    if key == (edges, mdl, sdiff):
                        c_edges = set(g_obj.get_edges())
                        shd = compute_shd(c_edges, gt_edges_set)
                        f1 = compute_f1(c_edges, gt_edges_set)
                        shd_s = compute_shd_skeleton(c_edges, gt_edges_set)
                        f1_s = compute_f1_skeleton(c_edges, gt_edges_set)
                        f.write(f"{i},{edges},{mdl:.2f},{sdiff},{shd},{f1:.4f},{shd_s},{f1_s:.4f},{count}\n")
                        break
            else:
                f.write(f"{i},{edges},{mdl:.2f},{sdiff},{count}\n")
    extra = ", SHD+F1" if gt_graph is not None else ""
    print(f"  Pareto CSV: {pareto_path} ({len(pf_counts)} 个唯一解{extra})")

    # 保存实验参数（batch 子进程跳过，由 _run_batch 统一输出）
    if not args.no_params:
        import json
        params_path = os.path.join(args.output, "params.json")
        params = {
            "prior_file": args.prior_file,
            "data_file": args.data_file,
            "ground_truth": args.ground_truth,
            "pop_size": args.pop_size,
            "generations": args.generations,
            "neighbors": args.neighbors,
            "prob_neighbor": args.prob_neighbor,
            "max_replace": args.max_replace,
            "crossover_type": args.crossover_type,
            "mutation_prob": args.mutation_prob,
            "mutation_ops_min": args.mutation_ops_min,
            "mutation_ops_max": args.mutation_ops_max,
            "max_sdiff": max_sdiff,
            "max_parents": args.max_parents,
            "mdl_penalty": args.mdl_penalty,
            "seed": args.seed,
            "git_commit": _get_git_commit(),
            "n_nodes": len(node_names),
            "n_states": n_states,
        }
        with open(params_path, "w") as f:
            json.dump(params, f, indent=2)
        print(f"  实验参数: {params_path}")

    # 可视化
    if not args.no_plot:
        print(f"  生成图表...")

        pareto_plot = os.path.join(args.output, "pareto_front.png")
        plot_pareto_front(result, save_path=pareto_plot, original_pos=gt_pos)

        conv_plot = os.path.join(args.output, "convergence.png")
        plot_convergence(result, save_path=conv_plot)

        obj_conv_plot = os.path.join(args.output, "objective_convergence.png")
        plot_objective_convergence(result, save_path=obj_conv_plot)

        if args.plot_networks and result.pareto_graphs and _has_networkx():
            _plot_three_networks(result, args.output, node_names)

    print(f"\n  全部完成! 输出目录: {args.output}")


# ═══════════════════════════════════════════════════════════════
# Batch 并行运行
# ═══════════════════════════════════════════════════════════════

def _batch_worker(seed: int, output_dir: str, prior_file: str,
                  data_file: str, args) -> str:
    """在子进程中执行一次单运行。"""
    cmd = [
        sys.executable, os.path.abspath(__file__),
        "--prior-file", prior_file,
        "--data-file", data_file,
        "--pop-size", str(args.pop_size),
        "--generations", str(args.generations),
        "--max-sdiff", str(args.max_sdiff),
        "--mdl-penalty", str(args.mdl_penalty),
        "--neighbors", str(args.neighbors),
        "--prob-neighbor", str(args.prob_neighbor),
        "--max-replace", str(args.max_replace),
        "--crossover-type", str(args.crossover_type),
        "--mutation-prob", str(args.mutation_prob),
        "--mutation-ops-min", str(args.mutation_ops_min),
        "--mutation-ops-max", str(args.mutation_ops_max),
        "--seed", str(seed),
        "--output", output_dir,
        "--no-plot",
        "--no-params",
    ]
    if args.plot_networks:
        cmd.append("--plot-networks")
    if args.max_parents is not None:
        cmd.append("--max-parents")
        cmd.append(str(args.max_parents))
    if args.ground_truth:
        cmd.extend(["--ground-truth", args.ground_truth])

    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return output_dir


def _run_batch(args):
    """并行多次 MOEA/D 运行，汇总 Pareto 前沿。"""
    os.makedirs(args.output, exist_ok=True)

    # 加载先验以计算 max_sdiff
    prior_graph, node_names, n_states, known_node_indices = _load_prior_file(args.prior_file)

    # 自动计算 max_sdiff 理论最大值
    if args.max_sdiff is None:
        n_prior_edges = len(prior_graph.get_edges())
        args.max_sdiff = _compute_max_sdiff(len(node_names), args.max_parents,
                                              n_prior_edges, known_node_indices)
        n_known = len(known_node_indices) if known_node_indices is not None else len(node_names)
        print(f"  max_sdiff 自动计算: {n_known}×{args.max_parents or (len(node_names)-1)}"
              f"+{n_prior_edges} = {args.max_sdiff}")

    seeds = list(range(42, 42 + args.batch))

    print(f"\n启动 {args.batch} 次并行运行 (workers={args.workers or 'auto'})...")
    print(f"参数: pop={args.pop_size}, gen={args.generations}, "
          f"max_sdiff={args.max_sdiff}, max_parents={args.max_parents}")

    # ── 并行执行 ──────────────────────────────────────────
    output_dirs = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for seed in seeds:
            out_dir = os.path.join(args.output, f"batch_{seed}")
            futures[executor.submit(
                _batch_worker, seed, out_dir, args.prior_file, args.data_file, args
            )] = seed

        for future in as_completed(futures):
            seed = futures[future]
            try:
                d = future.result()
                output_dirs.append(d)
                print(f"  [ok] seed={seed} ({len(output_dirs)}/{args.batch})")
            except subprocess.CalledProcessError as e:
                print(f"  [fail] seed={seed}: 返回码 {e.returncode}")

    print(f"\n完成 {len(output_dirs)}/{args.batch} 次运行")

    if not output_dirs:
        print("没有可用的结果，退出。")
        return

    # ── 收集结果 ──────────────────────────────────────────
    print("收集结果...")
    results = []
    for d in output_dirs:
        pkl_path = os.path.join(d, "result.pkl")
        if not os.path.exists(pkl_path):
            print(f"  [warn] 缺少: {pkl_path}")
            continue
        with open(pkl_path, "rb") as f:
            r = pickle.load(f)
        results.append({
            "name": os.path.basename(d),
            "pareto_f": r.pareto_f.copy(),
            "pareto_graphs": r.pareto_graphs,
            "runtime": r.runtime,
        })
        print(f"  {os.path.basename(d)}: {len(r.pareto_f)} 个 Pareto 解, "
              f"耗时 {r.runtime:.0f}s")

    # ── 汇总统计 ──────────────────────────────────────────
    runtimes = [r["runtime"] for r in results]
    pareto_sizes = [len(r["pareto_f"]) for r in results]
    print(f"\n统计 ({len(results)} 次运行):")
    print(f"  耗时: 均值 {np.mean(runtimes):.0f}s, "
          f"最小 {np.min(runtimes):.0f}s, 最大 {np.max(runtimes):.0f}s")
    print(f"  Pareto 解数: 均值 {np.mean(pareto_sizes):.1f}, "
          f"最小 {np.min(pareto_sizes)}, 最大 {np.max(pareto_sizes)}")

    # ── 保存实验参数 ──────────────────────────────────────
    import json
    model_label = args.model or os.path.splitext(os.path.basename(args.prior_file))[0]
    params = {
        "prior_file": args.prior_file,
        "data_file": args.data_file,
        "model": model_label,
        "pop_size": args.pop_size,
        "generations": args.generations,
        "neighbors": args.neighbors,
        "prob_neighbor": args.prob_neighbor,
        "max_replace": args.max_replace,
        "crossover_type": args.crossover_type,
        "mutation_prob": args.mutation_prob,
        "mutation_ops_min": args.mutation_ops_min,
        "mutation_ops_max": args.mutation_ops_max,
        "max_sdiff": args.max_sdiff,
        "max_parents": args.max_parents,
        "mdl_penalty": args.mdl_penalty,
        "git_commit": _get_git_commit(),
        "batch": args.batch,
        "workers": args.workers,
    }
    params_path = os.path.join(args.output, "params.json")
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)

    # ── 汇总绘图 ──────────────────────────────────────────
    _plot_batch_combined(results, args.output, model_label,
                         args.pop_size, args.generations)

    print(f"\n全部完成! 输出目录: {args.output}")
    print(f"  params.json — 实验参数")
    print(f"  combined_pareto.png — 全量汇总图")


# ═══════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════

def _get_git_commit() -> str:
    """获取当前代码的 git commit 短哈希。"""
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _compute_max_sdiff(n_nodes: int, max_parents: int | None,
                        prior_edges: int,
                        known_node_indices: list[int] | None = None) -> int:
    """计算 sdiff 的理论最大值。

    max_sdiff = n_known × K + E_prior_known
    其中 K = max_parents（若未限制则为 n_nodes - 1）。
    当 known_node_indices 为 None 时，全部节点已知。
    """
    K = max_parents if max_parents is not None else (n_nodes - 1)
    n_known = len(known_node_indices) if known_node_indices is not None else n_nodes
    return n_known * K + prior_edges


def _has_networkx() -> bool:
    """检查 networkx 是否可用。"""
    try:
        import networkx  # noqa: F401
        return True
    except ImportError:
        return False


def _plot_batch_combined(results: list[dict], output_dir: str, model: str,
                         pop_size: int, generations: int) -> None:
    """绘制 batch 全量汇总图（全部前沿点）。"""
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use("Agg")

    fig, ax = plt.subplots(figsize=(10, 8))
    n_results = len(results)

    if n_results <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, n_results))
    else:
        colors = plt.cm.tab20(np.arange(n_results) % 20 / 20.0)

    for i, r in enumerate(results):
        f = r["pareto_f"]
        ax.scatter(
            f[:, 0], f[:, 1],
            c=[colors[i % 20]], s=25, alpha=0.4,
            edgecolors="none", label=r["name"],
        )

    ax.set_xlabel("MDL Score", fontsize=13)
    ax.set_ylabel("Structural Symmetric Difference", fontsize=13)
    ax.set_title(
        f"Pareto Fronts — {n_results} runs "
        f"({model}, pop={pop_size}, gen={generations})",
        fontsize=14,
    )
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    save_path = os.path.join(output_dir, "combined_pareto.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"汇总图已保存: {save_path}")
    plt.close(fig)


def _plot_three_networks(result, output_dir: str, node_names: list[str]) -> None:
    """绘制三个代表解：最接近先验、MDL最优、两者平衡。"""
    f = result.pareto_f
    graphs = result.pareto_graphs
    n = len(graphs)

    idx_prior = int(np.argmin(f[:, 1]))
    idx_mdl = int(np.argmin(f[:, 0]))

    if n > 2:
        min_f = f.min(axis=0)
        max_f = f.max(axis=0)
        range_f = np.maximum(max_f - min_f, 1e-8)
        f_norm = (f - min_f) / range_f
        endpoints = {idx_prior, idx_mdl}
        mid = np.array([f_norm[idx_mdl, 0], f_norm[idx_prior, 1]])
        best_dist = float("inf")
        idx_bal = 0
        for i in range(n):
            if i in endpoints:
                continue
            d = np.sqrt((f_norm[i, 0] - mid[0]) ** 2
                        + (f_norm[i, 1] - mid[1]) ** 2)
            if d < best_dist:
                best_dist = d
                idx_bal = i
        if best_dist == float("inf"):
            idx_bal = idx_prior
    else:
        idx_bal = 0

    labels = [
        (idx_prior, "closest_to_prior", "Closest to Prior (min Sdiff)"),
        (idx_mdl, "best_mdl", "Best MDL"),
        (idx_bal, "balanced", "Balanced"),
    ]

    for idx, fname, title in labels:
        path = os.path.join(output_dir, f"network_{fname}.png")
        plot_network(graphs[idx], node_names, title=title, save_path=path)


if __name__ == "__main__":
    main()
