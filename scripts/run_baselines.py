#!/usr/bin/env python3
"""对照组结构学习实验：通过 R bnlearn 运行 HC, Tabu, MMHC, PC-stable, inter-IAMB。

用法:
    python scripts/run_baselines.py                          # 全部 6 网络 × 4 样本量
    python scripts/run_baselines.py --networks asia,alarm    # 指定网络
    python scripts/run_baselines.py --samples 500,5000       # 指定样本量
    python scripts/run_baselines.py --dry-run                # 仅列出任务，不执行

前置条件: R + bnlearn 已安装 (bash scripts/install_r.sh)
输出: output/baselines/results.csv
"""

import csv
import os
import pickle
import subprocess
import sys
import tempfile
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.graph import DirectedGraph
from src.metrics import compute_f1, compute_shd
from src.score import MDLScore

R_SCRIPT = os.path.join(ROOT, "scripts", "baseline_bnlearn.R")
R_LIBS = os.path.join(ROOT, "venv", "R_libs")

# ── 配置 ──────────────────────────────────────────────────────────────────

NETWORKS = ["asia", "alarm", "hailfinder", "win95pts", "munin1", "andes"]
SAMPLE_SIZES = [500, 1000, 5000, 10000]

DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "output", "baselines")


# ── 工具函数 ──────────────────────────────────────────────────────────────

def load_task(network: str, n_samples: int):
    """加载数据、GT 和元信息。返回 (data_np, gt_edges, n_states, node_names)。"""
    if network not in NETWORKS:
        raise ValueError(f"未知网络: {network}，合法值: {NETWORKS}")
    gt_path = os.path.join(DATA_DIR, "ground_truth", f"{network}_graph.pkl")
    with open(gt_path, "rb") as f:
        gt_graph, node_names, n_states = pickle.load(f)
    gt_edges = set(gt_graph.get_edges())
    data_path = os.path.join(DATA_DIR, "synthetic", f"{network}_N{n_samples}.npy")
    data_np = np.load(data_path).astype(np.int32)
    return data_np, gt_edges, n_states, node_names


def run_one(network: str, n_samples: int) -> list[dict]:
    """调用 R bnlearn 跑 5 个算法，返回结果 list。"""
    data_np, gt_edges, n_states, node_names = load_task(network, n_samples)
    name2idx = {n: i for i, n in enumerate(node_names)}
    n_nodes = len(node_names)
    mdl_scorer = MDLScore(data_np, n_states, penalty_scale=1.0)

    # 写临时 CSV → R
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", prefix=f"bnlearn_in_{network}_N{n_samples}_", delete=False
    ) as f:
        pd.DataFrame(data_np, columns=node_names).to_csv(f, index=False)
        input_csv = f.name

    output_csv = input_csv.replace("_in_", "_out_")
    env = {**os.environ, "R_LIBS_USER": R_LIBS}
    algo_names = ["HC", "Tabu", "MMHC", "PC-stable", "inter-IAMB"]
    try:
        subprocess.run(
            ["Rscript", R_SCRIPT, input_csv, output_csv],
            capture_output=True, text=True, check=True, timeout=600, env=env,
        )
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {network} N={n_samples}", file=sys.stderr)
        return [_empty_row(network, n_samples, a) for a in algo_names]
    except subprocess.CalledProcessError as e:
        print(f"  [R ERROR] {network} N={n_samples}: {e.stderr.strip()}", file=sys.stderr)
        return [_empty_row(network, n_samples, a) for a in algo_names]
    finally:
        os.unlink(input_csv)

    # 读回边表
    try:
        edges_df = pd.read_csv(output_csv)
    except Exception:
        return [_empty_row(network, n_samples, a) for a in algo_names]
    finally:
        if os.path.exists(output_csv):
            os.unlink(output_csv)

    results = []
    for algo, group in edges_df.groupby("algorithm"):
        edge_set = set(
            (name2idx[r["from"]], name2idx[r["to"]])
            for _, r in group.iterrows() if r["from"] != "" and r["to"] != ""
        )
        runtime = float(group["runtime_sec"].iloc[0])
        g = DirectedGraph.from_edges(n_nodes, list(edge_set))
        results.append(_make_row(
            network, n_samples, _algo_label(algo), g, edge_set, gt_edges,
            mdl_scorer, runtime,
        ))
    return results


def _algo_label(name: str) -> str:
    """R 脚本中的短名 → CSV 中的人读标签。"""
    return {"hc": "HC", "tabu": "Tabu", "mmhc": "MMHC",
            "pc": "PC-stable", "iamb": "inter-IAMB"}.get(name, name)


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
    parser = ArgumentParser(description="对照组结构学习实验 (R bnlearn)")
    parser.add_argument("--networks", type=str, default=None,
                        help=f"逗号分隔，默认: {','.join(NETWORKS)}")
    parser.add_argument("--samples", type=str, default=None,
                        help=f"逗号分隔，默认: {','.join(map(str, SAMPLE_SIZES))}")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅列出任务，不执行")
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                        help="并行 worker 数，默认: CPU 核心数")
    args = parser.parse_args()

    networks = args.networks.split(",") if args.networks else NETWORKS
    samples = [int(s) for s in args.samples.split(",")] if args.samples else SAMPLE_SIZES

    tasks = [(n, s) for n in networks for s in samples]

    print(f"对照组实验 (R bnlearn): {len(tasks)} 个任务 × 5 个算法 = {len(tasks) * 5} 次运行")
    print(f"网络: {networks}")
    print(f"样本量: {samples}")
    print(f"并行: {args.workers} workers")
    print(f"R 脚本: {R_SCRIPT}")
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
                n_ok = sum(1 for r in rows if r["edges"] is not None)
                print(f"[{total - len(futures)}/{total}] {network} N={n_samples} ({n_ok}/5 ok)")

    print(f"\n结果已写入 {csv_path}")


if __name__ == "__main__":
    main()
