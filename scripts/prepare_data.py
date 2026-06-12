#!/usr/bin/env python3
"""预生成实验数据集：ground truth、合成数据、先验网络。

用法:
    python scripts/prepare_data.py                          # 全部生成
    python scripts/prepare_data.py --networks asia,alarm    # 指定网络
    python scripts/prepare_data.py --samples 500,5000       # 指定样本量

所有文件存入 data/ 目录（已 gitignore）。
"""

import json
import math
import os
import pickle
import random
import sys
from argparse import ArgumentParser

import numpy as np

# 确保项目根在 path 中
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.prior import PriorNetwork, compute_n_perturb
from src.graph import DirectedGraph

# ── 配置 ──────────────────────────────────────────────────────────────

NETWORKS = ["asia", "alarm", "hailfinder", "win95pts", "munin1", "andes"]
SAMPLE_SIZES = [500, 1000, 5000, 10000]
COVERAGE_PCTS = [0.3, 0.5, 0.7, 1.0]  # 不含 empty（单独处理）
SEED_DATA = 9999
SEED_NODE_SAMPLE = 9999
SEED_PERTURB = 9999

DATA_DIR = os.path.join(ROOT, "data")


# ── Ground Truth ──────────────────────────────────────────────────────

def generate_ground_truth(network: str):
    """保存 ground truth 网络结构和元信息。"""
    out_dir = os.path.join(DATA_DIR, "ground_truth")
    os.makedirs(out_dir, exist_ok=True)

    graph, node_names, n_states = PriorNetwork.from_pgmpy_model(network)
    edges = graph.get_edges()
    n_edges = len(edges)

    # 保存图
    graph_path = os.path.join(out_dir, f"{network}_graph.pkl")
    with open(graph_path, "wb") as f:
        pickle.dump((graph, node_names, n_states), f)

    # 保存元信息 (JSON, 方便人类查看)
    info = {
        "name": network,
        "n_nodes": len(node_names),
        "n_edges": n_edges,
        "n_states": n_states,
        "node_names": node_names,
        "edges": [(node_names[u], node_names[v]) for u, v in edges],
    }
    info_path = os.path.join(out_dir, f"{network}_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f"  [GT] {network}: {len(node_names)} 节点, {n_edges} 边 → {graph_path}")
    return graph, node_names, n_states


# ── 合成数据 ──────────────────────────────────────────────────────────

def generate_synthetic_data(network: str, n_samples: int):
    """从标准网络采样合成数据。"""
    out_dir = os.path.join(DATA_DIR, "synthetic")
    os.makedirs(out_dir, exist_ok=True)

    data, node_names, n_states = PriorNetwork.generate_data(
        network, n_samples=n_samples, seed=SEED_DATA
    )

    # 保存数据
    npy_path = os.path.join(out_dir, f"{network}_N{n_samples}.npy")
    np.save(npy_path, data)

    # 保存元信息
    meta = {
        "network": network,
        "n_samples": n_samples,
        "n_nodes": len(node_names),
        "node_names": node_names,
        "n_states": n_states,
        "seed": SEED_DATA,
    }
    meta_path = os.path.join(out_dir, f"{network}_N{n_samples}.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"  [DATA] {network} N={n_samples}: {data.shape} → {npy_path}")
    return data


# ── 先验网络 ──────────────────────────────────────────────────────────

def _sample_nodes(n_nodes: int, pct: float, rng: random.Random) -> list[int]:
    """随机抽取 ceil(n_nodes * pct) 个节点索引。"""
    k = max(1, math.ceil(n_nodes * pct))
    return sorted(rng.sample(range(n_nodes), k))


def _extract_subgraph_edges(
    graph: DirectedGraph, known_indices: list[int]
) -> list[tuple[int, int]]:
    """提取已知节点之间的所有边。"""
    known_set = set(known_indices)
    edges = []
    for u, v in graph.get_edges():
        if u in known_set and v in known_set:
            edges.append((u, v))
    return edges


def generate_empty_prior(n_nodes: int, node_names: list[str], n_states: list[int]):
    """生成空图先验（全节点，零边）。"""
    graph = DirectedGraph.from_edges(n_nodes, [])
    prior_data = {
        "graph": graph,
        "node_names": node_names,
        "n_states": n_states,
        "prior_type": "empty",
    }
    return prior_data


def generate_coverage_prior(
    ground_truth: DirectedGraph,
    node_names: list[str],
    n_states: list[int],
    coverage_pct: float,
):
    """生成覆盖率型先验。

    1. 随机抽取 k% 节点
    2. 保留这些节点间的边
    3. 应用 perturb
    """
    rng_nodes = random.Random(SEED_NODE_SAMPLE)

    n_nodes = len(node_names)
    known_indices = _sample_nodes(n_nodes, coverage_pct, rng_nodes)
    sub_edges = _extract_subgraph_edges(ground_truth, known_indices)

    # 构建子图（全节点，但仅已知节点间有边）
    graph = DirectedGraph.from_edges(n_nodes, sub_edges)

    # 扰动
    n_perturb = compute_n_perturb(len(sub_edges))
    if n_perturb > 0:
        graph = PriorNetwork.perturb(graph, n_changes=n_perturb, seed=SEED_PERTURB)

    prior_data = {
        "graph": graph,
        "node_names": node_names,
        "n_states": n_states,
        "coverage_pct": coverage_pct,
        "known_node_indices": known_indices,
    }
    return prior_data


def save_prior(network: str, prior_data: dict, suffix: str):
    """保存先验网络文件。"""
    out_dir = os.path.join(DATA_DIR, "priors")
    os.makedirs(out_dir, exist_ok=True)

    path = os.path.join(out_dir, f"{network}_{suffix}.pkl")
    with open(path, "wb") as f:
        pickle.dump(prior_data, f)

    graph = prior_data["graph"]
    n_edges = len(graph.get_edges())
    if "prior_type" in prior_data:
        detail = f"empty: {n_edges} 边"
    else:
        pct = prior_data["coverage_pct"]
        detail = f"pct={pct:.0%}: {len(prior_data['known_node_indices'])} 节点, {n_edges} 边"
    print(f"  [PRIOR] {network} {detail} → {path}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = ArgumentParser(description="预生成实验数据集")
    parser.add_argument(
        "--networks",
        type=str,
        default=None,
        help=f"逗号分隔的网络名，默认: {','.join(NETWORKS)}",
    )
    parser.add_argument(
        "--samples",
        type=str,
        default=None,
        help=f"逗号分隔的样本量，默认: {','.join(map(str, SAMPLE_SIZES))}",
    )
    args = parser.parse_args()

    networks = args.networks.split(",") if args.networks else NETWORKS
    samples = [int(s) for s in args.samples.split(",")] if args.samples else SAMPLE_SIZES

    print("=" * 60)
    print("准备实验数据集")
    print(f"  网络: {networks}")
    print(f"  样本量: {samples}")
    print(f"  数据 seed: {SEED_DATA}")
    print(f"  节点采样 seed: {SEED_NODE_SAMPLE}")
    print(f"  扰动 seed: {SEED_PERTURB}")
    print("=" * 60)

    for network in networks:
        print(f"\n── {network} ──")

        # 1. Ground truth
        gt_graph, node_names, n_states = generate_ground_truth(network)

        # 2. 合成数据
        for n in samples:
            generate_synthetic_data(network, n)

        # 3. 先验网络
        n_nodes = len(node_names)

        # 3a. 空图先验
        empty_prior = generate_empty_prior(n_nodes, node_names, n_states)
        save_prior(network, empty_prior, "empty")

        # 3b. 覆盖率先验
        for pct in COVERAGE_PCTS:
            cov_prior = generate_coverage_prior(gt_graph, node_names, n_states, pct)
            suffix = f"pct{int(pct * 100):03d}"
            save_prior(network, cov_prior, suffix)

    print(f"\n{'=' * 60}")
    print(f"完成。所有文件已写入 {DATA_DIR}/")
    print(f"  目录: {sorted(os.listdir(DATA_DIR))}")


if __name__ == "__main__":
    main()
