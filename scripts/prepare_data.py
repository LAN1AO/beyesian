#!/usr/bin/env python3
"""预生成实验数据集：ground truth、合成数据、先验网络。

用法:
    python scripts/prepare_data.py                          # 全部生成
    python scripts/prepare_data.py --networks asia,alarm    # 指定网络
    python scripts/prepare_data.py --samples 500,5000       # 指定样本量

所有文件存入 data/ 目录（已 gitignore）。
"""

import json
import os
import pickle
import sys
from argparse import ArgumentParser

import numpy as np

# 确保项目根在 path 中
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.prior import PriorNetwork

# ── 配置 ──────────────────────────────────────────────────────────────

NETWORKS = ["asia", "alarm", "hailfinder", "win95pts", "munin1", "andes"]
SAMPLE_SIZES = [500, 1000, 5000, 10000]
PERTURBATION_LEVELS = {"mild": 0.10, "moderate": 0.25, "severe": 0.50}
MAX_PARENTS = {
    "asia": 4,
    "alarm": 6, "hailfinder": 6,
    "win95pts": 8, "munin1": 8, "andes": 8,
}
SEED_DATA = 9999
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

def save_prior(network: str, prior_data: dict, suffix: str):
    """保存先验网络文件。"""
    out_dir = os.path.join(DATA_DIR, "priors")
    os.makedirs(out_dir, exist_ok=True)

    path = os.path.join(out_dir, f"{network}_{suffix}.pkl")
    with open(path, "wb") as f:
        pickle.dump(prior_data, f)

    graph = prior_data["graph"]
    n_edges = len(graph.get_edges())
    ptype = prior_data["prior_type"]
    shd = prior_data.get("shd_from_gt", 0)
    print(f"  [PRIOR] {network} {ptype}: {n_edges} 边, SHD={shd} → {path}")


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
    print(f"  扰动级别: {list(PERTURBATION_LEVELS.keys())} + random")
    print(f"  数据 seed: {SEED_DATA}, 扰动 seed: {SEED_PERTURB}")
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
        n_edges = len(gt_graph.get_edges())
        mp = MAX_PARENTS[network]
        density = n_edges / (n_nodes * (n_nodes - 1) / 2)

        # 3a. GT 先验（完美先验）
        gt_prior = {
            "graph": gt_graph.copy(),
            "node_names": node_names,
            "n_states": n_states,
            "prior_type": "gt",
            "shd_from_gt": 0,
        }
        save_prior(network, gt_prior, "gt")

        # 3b. 扰动先验（mild / moderate / severe）
        for label, frac in PERTURBATION_LEVELS.items():
            g, meta = PriorNetwork.construct_perturbed(
                gt_graph, frac, seed=SEED_PERTURB, max_parents=mp,
            )
            prior_data = {
                "graph": g,
                "node_names": node_names,
                "n_states": n_states,
                "prior_type": "perturbed",
                "delete_frac": frac,
                **meta,
            }
            save_prior(network, prior_data, label)

        # 3c. 随机 DAG 先验
        rg = PriorNetwork.random_dag(n_nodes, density, seed=SEED_PERTURB,
                                     max_parents=mp)
        gt_edges = set(gt_graph.get_edges())
        rg_edges = set(rg.get_edges())
        random_prior = {
            "graph": rg,
            "node_names": node_names,
            "n_states": n_states,
            "prior_type": "random",
            "density": round(density, 6),
            "shd_from_gt": len(rg_edges ^ gt_edges),
        }
        save_prior(network, random_prior, "random")

    print(f"\n{'=' * 60}")
    print(f"完成。所有文件已写入 {DATA_DIR}/")
    print(f"  目录: {sorted(os.listdir(DATA_DIR))}")


if __name__ == "__main__":
    main()
