"""多目标贝叶斯网络结构学习 — CLI 入口。

使用 MOEA/D + 切比雪夫分解，优化 MDL 评分和结构对称差。

单次运行:
  python3 main.py --model alarm --pop-size 100 --generations 500

并行 batch 运行 (N=20):
  python3 main.py --model alarm --batch 20 --pop-size 100 --generations 10000
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
from src.moead import MOEAD
from src.prior import PriorNetwork
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

    # 先验网络
    parser.add_argument(
        "--model", type=str, default="asia",
        help="bnlearn 网络名 (默认: asia)",
    )
    parser.add_argument(
        "--bif", type=str, default=None,
        help="BIF 文件路径 (优先级高于 --model)",
    )

    # 数据
    parser.add_argument(
        "--n-samples", type=int, default=500,
        help="合成数据样本数 (默认: 500)",
    )
    parser.add_argument(
        "--data-file", type=str, default=None,
        help="预生成数据文件 (.npy)，用于 batch 共享数据",
    )
    parser.add_argument(
        "--prior-file", type=str, default=None,
        help="预生成先验网络文件 (.pkl)，用于 batch 共享先验",
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
        "--max-sdiff", type=int, default=15,
        help="结构对称差上限 (默认: 15)",
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

def _run_single(args):
    """单次 MOEA/D 运行。"""
    os.makedirs(args.output, exist_ok=True)

    # ── 1. 加载先验网络 ──────────────────────────────────────
    print(f"[1/5] 加载先验网络...")
    original_graph = None
    if args.prior_file:
        with open(args.prior_file, "rb") as f:
            prior_graph, node_names, n_states = pickle.load(f)
        print(f"  从文件加载先验: {args.prior_file}")
    elif args.bif:
        prior_graph, node_names, n_states = PriorNetwork.from_bif(args.bif)
    else:
        prior_graph, node_names, n_states = PriorNetwork.from_pgmpy_model(args.model)
        original_graph = prior_graph.copy()
        prior_graph = PriorNetwork.perturb(prior_graph, n_changes=6, seed=args.seed)
        print(f"  原始边数: {len(original_graph.get_edges())}, "
              f"变异后边数: {len(prior_graph.get_edges())}")
    print(f"  网络: {args.model}, 节点: {len(node_names)}")

    # ── 2. 准备数据 ──────────────────────────────────────────
    print(f"[2/5] 准备数据...")
    if args.data_file:
        data = np.load(args.data_file).astype(np.int32)
        print(f"  从文件加载: {data.shape}")
    else:
        data, _, _ = PriorNetwork.generate_data(
            args.model, n_samples=args.n_samples, seed=args.seed
        )
        print(f"  合成数据: {data.shape}, 样本数: {args.n_samples}")

    # ── 3. 配置 MOEA/D ───────────────────────────────────────
    print(f"[3/5] 配置 MOEA/D...")
    config = MOEADConfig(
        n_nodes=len(node_names),
        n_states=n_states,
        max_parents=args.max_parents,
        mdl_penalty_scale=args.mdl_penalty,
        max_symmetric_diff=args.max_sdiff,
        n_weight_vectors=args.pop_size,
        n_neighbors=args.neighbors,
        n_generations=args.generations,
        prob_neighbor_mating=args.prob_neighbor,
        max_replacements=args.max_replace,
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

    # 保存 Pareto 前沿 CSV（合并重复解，标记数量）
    pareto_path = os.path.join(args.output, "pareto_front.csv")
    with open(pareto_path, "w") as f:
        f.write("index,edges,mdl,sdiff,count\n")
        for i, ((edges, mdl, sdiff), count) in enumerate(
            sorted(pf_counts.items(), key=lambda x: (x[0][2], x[0][1]))
        ):
            f.write(f"{i},{edges},{mdl:.2f},{sdiff},{count}\n")
    print(f"  Pareto CSV: {pareto_path} ({len(pf_counts)} 个唯一解)")

    # 保存实验参数（batch 子进程跳过，由 _run_batch 统一输出）
    if not args.no_params:
        import json
        params_path = os.path.join(args.output, "params.json")
        params = {
            "model": args.model,
            "bif": args.bif,
            "n_samples": args.n_samples,
            "pop_size": args.pop_size,
            "generations": args.generations,
            "neighbors": args.neighbors,
            "prob_neighbor": args.prob_neighbor,
            "max_replace": args.max_replace,
            "mutation_prob": args.mutation_prob,
            "mutation_ops_min": args.mutation_ops_min,
            "mutation_ops_max": args.mutation_ops_max,
            "max_sdiff": args.max_sdiff,
            "max_parents": args.max_parents,
            "mdl_penalty": args.mdl_penalty,
            "seed": args.seed,
            "n_nodes": len(node_names),
            "n_states": n_states,
        }
        with open(params_path, "w") as f:
            json.dump(params, f, indent=2)
        print(f"  实验参数: {params_path}")

    # 可视化
    if not args.no_plot:
        print(f"  生成图表...")

        original_pos = None
        if original_graph is not None:
            from src.score import MDLScore, StructuralDiffScore
            mdl_scorer = MDLScore(data, n_states,
                                  penalty_scale=args.mdl_penalty)
            sdiff_scorer = StructuralDiffScore(prior_graph)
            orig_mdl = mdl_scorer.score_graph(original_graph)
            orig_sdiff = sdiff_scorer.score_graph(original_graph)
            original_pos = (orig_mdl, orig_sdiff)
            print(f"  原始图目标值: MDL={orig_mdl:.2f}, Sdiff={orig_sdiff:.0f}")

        pareto_plot = os.path.join(args.output, "pareto_front.png")
        plot_pareto_front(result, save_path=pareto_plot, original_pos=original_pos)

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
        "--mutation-prob", str(args.mutation_prob),
        "--mutation-ops-min", str(args.mutation_ops_min),
        "--mutation-ops-max", str(args.mutation_ops_max),
        "--model", args.model,
        "--seed", str(seed),
        "--output", output_dir,
        "--no-plot",
        "--no-params",
    ]
    if args.bif:
        cmd.extend(["--bif", args.bif])
    if args.plot_networks:
        cmd.append("--plot-networks")
    if args.max_parents is not None:
        cmd.append("--max-parents")
        cmd.append(str(args.max_parents))

    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return output_dir


def _run_batch(args):
    """并行多次 MOEA/D 运行，汇总 Pareto 前沿。"""
    os.makedirs(args.output, exist_ok=True)

    # ── 预生成共享先验网络和数据 ─────────────────────────
    shared_dir = os.path.join(args.output, "shared")
    prior_file = os.path.join(shared_dir, "prior.pkl")
    data_file = os.path.join(shared_dir, "data.npy")

    if not (os.path.exists(prior_file) and os.path.exists(data_file)):
        print("预生成共享先验网络和数据...")
        os.makedirs(shared_dir, exist_ok=True)

        if args.bif:
            prior_graph, node_names, n_states = PriorNetwork.from_bif(args.bif)
        else:
            prior_graph, node_names, n_states = PriorNetwork.from_pgmpy_model(args.model)
            prior_graph = PriorNetwork.perturb(prior_graph, n_changes=6, seed=9999)
        with open(prior_file, "wb") as f:
            pickle.dump((prior_graph, node_names, n_states), f)
        print(f"  共享先验: {len(node_names)} 节点, "
              f"{len(prior_graph.get_edges())} 边")

        if args.bif:
            from pgmpy.readwrite import BIFReader
            model = BIFReader(args.bif).get_model()
            data, _, _ = PriorNetwork.generate_data(
                model, n_samples=args.n_samples, seed=9999,
            )
        else:
            data, _, _ = PriorNetwork.generate_data(
                args.model, n_samples=args.n_samples, seed=9999,
            )
        np.save(data_file, data)
        print(f"  共享数据: {data.shape[0]} 样本")
    else:
        print(f"共享文件已存在，跳过生成: {shared_dir}/")

    seeds = list(range(42, 42 + args.batch))

    print(f"\n启动 {args.batch} 次并行运行 (workers={args.workers or 'auto'})...")
    print(f"参数: model={args.model}, pop={args.pop_size}, "
          f"gen={args.generations}, samples={args.n_samples}, "
          f"max_sdiff={args.max_sdiff}, max_parents={args.max_parents}")

    # ── 并行执行 ──────────────────────────────────────────
    output_dirs = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for seed in seeds:
            out_dir = os.path.join(args.output, f"batch_{seed}")
            futures[executor.submit(
                _batch_worker, seed, out_dir, prior_file, data_file, args
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
    params = {
        "model": args.model,
        "bif": args.bif,
        "n_samples": args.n_samples,
        "pop_size": args.pop_size,
        "generations": args.generations,
        "neighbors": args.neighbors,
        "prob_neighbor": args.prob_neighbor,
        "max_replace": args.max_replace,
        "mutation_prob": args.mutation_prob,
        "mutation_ops_min": args.mutation_ops_min,
        "mutation_ops_max": args.mutation_ops_max,
        "max_sdiff": args.max_sdiff,
        "max_parents": args.max_parents,
        "mdl_penalty": args.mdl_penalty,
        "batch": args.batch,
        "workers": args.workers,
    }
    params_path = os.path.join(args.output, "params.json")
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)

    # ── 汇总绘图 ──────────────────────────────────────────
    _plot_batch_combined(results, args.output, args.model,
                         args.pop_size, args.generations)

    print(f"\n全部完成! 输出目录: {args.output}")
    print(f"  params.json — 实验参数")
    print(f"  combined_pareto.png — 全量汇总图")


# ═══════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════

def _peek_n_nodes(bif_path: str | None, model_name: str,
                   prior_file: str | None = None) -> int:
    """快速获取模型的节点数。"""
    if prior_file:
        with open(prior_file, "rb") as f:
            _, names, _ = pickle.load(f)
        return len(names)
    if bif_path:
        from pgmpy.readwrite import BIFReader
        return len(BIFReader(bif_path).get_model().nodes())
    else:
        from pgmpy.example_models import load_model
        name = model_name
        if not name.startswith("bnlearn/") and not name.startswith("bnrep/"):
            name = f"bnlearn/{name}"
        return len(load_model(name).nodes())


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
