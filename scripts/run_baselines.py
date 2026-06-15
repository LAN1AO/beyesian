#!/usr/bin/env python3
"""对照组结构学习实验：HC, Tabu, GES, MMHC, PC-stable 在全部数据集上运行。

用法:
    python scripts/run_baselines.py                          # 全部 6 网络 × 4 样本量
    python scripts/run_baselines.py --networks asia,alarm    # 指定网络
    python scripts/run_baselines.py --samples 500,5000       # 指定样本量
    python scripts/run_baselines.py --dry-run                # 仅列出任务，不执行

输出: output/baselines/results.csv
"""

import csv
import os
import pickle
import sys
import time
import warnings
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

# 抑制 pgmpy 的 deprecation warning
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pgmpy import config as pgmpy_config
from pgmpy.estimators import BIC, GES, PC, HillClimbSearch, MmhcEstimator

from src.graph import DirectedGraph
from src.metrics import compute_f1, compute_shd
from src.score import MDLScore

pgmpy_config.set_show_progress(False)

# ── 配置 ──────────────────────────────────────────────────────────────────

NETWORKS = ["asia", "alarm", "hailfinder", "win95pts", "munin1", "andes"]
SAMPLE_SIZES = [500, 1000, 5000, 10000]

DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "output", "baselines")


# ── 工具函数 ──────────────────────────────────────────────────────────────

def load_task(network: str, n_samples: int):
    """加载数据、GT 和元信息。返回 (df, gt_edges, n_states, node_names, data_np)。"""
    if network not in NETWORKS:
        raise ValueError(f"未知网络: {network}，合法值: {NETWORKS}")
    # GT
    gt_path = os.path.join(DATA_DIR, "ground_truth", f"{network}_graph.pkl")
    with open(gt_path, "rb") as f:
        gt_graph, node_names, n_states = pickle.load(f)
    gt_edges = set(gt_graph.get_edges())

    # Data
    data_path = os.path.join(DATA_DIR, "synthetic", f"{network}_N{n_samples}.npy")
    data_np = np.load(data_path).astype(np.int32)
    df = pd.DataFrame(data_np, columns=node_names)

    return df, gt_edges, n_states, node_names, data_np


def dag_to_edges(dag, name2idx: dict) -> set[tuple[int, int]]:
    """pgmpy DAG → int-index 边集。"""
    return set((name2idx[u], name2idx[v]) for u, v in dag.edges())


def run_one(network: str, n_samples: int) -> list[dict]:
    """在单个 (网络, 样本量) 上跑全部 5 个算法，返回结果 list。"""
    df, gt_edges, n_states, node_names, data_np = load_task(network, n_samples)
    name2idx = {n: i for i, n in enumerate(node_names)}
    n_nodes = len(node_names)
    mdl_scorer = MDLScore(data_np, n_states, penalty_scale=1.0)

    results = []
    bic = BIC(df)

    # ── HC ──
    t0 = time.time()
    dag = HillClimbSearch(df).estimate(
        scoring_method=bic, tabu_length=0, show_progress=False
    )
    edges = dag_to_edges(dag, name2idx)
    g = DirectedGraph.from_edges(n_nodes, list(edges))
    results.append(_make_row(network, n_samples, "HC", g, edges, gt_edges,
                             mdl_scorer, time.time() - t0))

    # ── Tabu ──
    t0 = time.time()
    dag = HillClimbSearch(df).estimate(
        scoring_method=bic, tabu_length=100, show_progress=False
    )
    edges = dag_to_edges(dag, name2idx)
    g = DirectedGraph.from_edges(n_nodes, list(edges))
    results.append(_make_row(network, n_samples, "Tabu", g, edges, gt_edges,
                             mdl_scorer, time.time() - t0))

    # ── GES ──
    t0 = time.time()
    try:
        pdag = GES(df).estimate(scoring_method=bic)
        dag = pdag.to_dag()
    except Exception as e:
        print(f"  [WARN] GES failed on {network} N={n_samples}: {e}", file=sys.stderr)
        results.append(_empty_row(network, n_samples, "GES"))
        dag = None
    if dag is not None:
        edges = dag_to_edges(dag, name2idx)
        g = DirectedGraph.from_edges(n_nodes, list(edges))
        results.append(_make_row(network, n_samples, "GES", g, edges, gt_edges,
                                 mdl_scorer, time.time() - t0))

    # ── MMHC ──
    t0 = time.time()
    dag = MmhcEstimator(df).estimate(scoring_method=bic)
    edges = dag_to_edges(dag, name2idx)
    g = DirectedGraph.from_edges(n_nodes, list(edges))
    results.append(_make_row(network, n_samples, "MMHC", g, edges, gt_edges,
                             mdl_scorer, time.time() - t0))

    # ── PC-stable ──
    t0 = time.time()
    dag = PC(df).estimate(
        variant="stable", ci_test="chi_square",
        return_type="dag", show_progress=False,
    )
    edges = dag_to_edges(dag, name2idx)
    g = DirectedGraph.from_edges(n_nodes, list(edges))
    results.append(_make_row(network, n_samples, "PC-stable", g, edges, gt_edges,
                             mdl_scorer, time.time() - t0))

    return results


def _make_row(network, n_samples, algo, graph, edges, gt_edges, mdl_scorer, runtime):
    return {
        "network": network,
        "n_samples": n_samples,
        "algorithm": algo,
        "edges": len(edges),
        "mdl": round(mdl_scorer.score_graph(graph), 2),
        "shd": compute_shd(edges, gt_edges),
        "f1": round(compute_f1(edges, gt_edges), 4),
        "runtime_sec": round(runtime, 1),
    }


def _empty_row(network, n_samples, algo):
    return {
        "network": network, "n_samples": n_samples, "algorithm": algo,
        "edges": None, "mdl": None, "shd": None, "f1": None, "runtime_sec": None,
    }


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = ArgumentParser(description="对照组结构学习实验")
    parser.add_argument("--networks", type=str, default=None,
                        help=f"逗号分隔，默认: {','.join(NETWORKS)}")
    parser.add_argument("--samples", type=str, default=None,
                        help=f"逗号分隔，默认: {','.join(map(str, SAMPLE_SIZES))}")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅列出任务，不执行")
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                        help=f"并行 worker 数，默认: CPU 核心数")
    args = parser.parse_args()

    networks = args.networks.split(",") if args.networks else NETWORKS
    samples = [int(s) for s in args.samples.split(",")] if args.samples else SAMPLE_SIZES

    tasks = [(n, s) for n in networks for s in samples]

    print(f"对照组实验: {len(tasks)} 个任务 × 5 个算法 = {len(tasks) * 5} 次运行")
    print(f"网络: {networks}")
    print(f"样本量: {samples}")
    print(f"并行: {args.workers} workers")
    print()

    if args.dry_run:
        for n, s in tasks:
            print(f"  {n} N={s}")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "results.csv")
    fieldnames = ["network", "n_samples", "algorithm", "edges", "mdl", "shd", "f1", "runtime_sec"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        total = len(tasks)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_one, n, s): (n, s) for n, s in tasks}
            for future in as_completed(futures):
                network, n_samples = futures[future]
                rows = future.result()
                writer.writerows(rows)
                f.flush()
                done = total - len(futures) + 1  # approx, good enough for progress
                n_ok = sum(1 for r in rows if r["edges"] is not None)
                print(f"[{done}/{total}] {network} N={n_samples} ({n_ok}/5 ok)")

    print(f"\n结果已写入 {csv_path}")


if __name__ == "__main__":
    main()
