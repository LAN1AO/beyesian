#!/usr/bin/env python3
"""全量 MOEA/D 实验运行脚本。

实验矩阵: 6 网络 × 5 先验 × 4 样本量 × 30 重复 = 3600 次运行。
使用 ThreadPoolExecutor 扁平化管理所有任务，全核满载。

用法:
    python scripts/run_experiments.py                                    # 全量运行
    python scripts/run_experiments.py --networks asia --priors gt,mild   # 子集
    python scripts/run_experiments.py --workers 64                       # 指定并行度
    python scripts/run_experiments.py --summary-only                     # 仅汇总已有结果

断点续跑: 若 run_{seed}/result.pkl 已存在则自动跳过。
"""

import csv
import os
import pickle
import subprocess
import sys
import time
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
MAIN_PY = os.path.join(ROOT, "main.py")

# ── 实验配置 ──────────────────────────────────────────────────────────

NETWORKS = ["asia", "alarm", "hailfinder", "win95pts", "munin1", "andes"]
PRIORS = ["gt", "mild", "moderate", "severe", "random"]
SAMPLE_SIZES = [500, 1000, 5000, 10000]
SEEDS = list(range(42, 72))  # 30 seeds

POP_SIZE = {
    "asia": 100, "alarm": 100, "hailfinder": 100,
    "win95pts": 200, "munin1": 200, "andes": 200,
}
MAX_PARENTS = {
    "asia": 4, "alarm": 6, "hailfinder": 6,
    "win95pts": 8, "munin1": 8, "andes": 8,
}
GENERATIONS = 10000

DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "output", "experiments")


# ── Worker ────────────────────────────────────────────────────────────

def run_single(network: str, prior: str, n_samples: int, seed: int) -> tuple:
    """运行一次 MOEA/D 实验。返回 (network, prior, n_samples, seed, status)。"""
    output_dir = os.path.join(
        OUT_DIR, f"{network}_{prior}_N{n_samples}", f"run_{seed}",
    )
    result_pkl = os.path.join(output_dir, "result.pkl")

    # 断点续跑：已有结果则跳过
    if os.path.exists(result_pkl):
        return (network, prior, n_samples, seed, "skipped")

    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        sys.executable, MAIN_PY,
        "--prior-file", os.path.join(DATA_DIR, "priors", f"{network}_{prior}.pkl"),
        "--data-file", os.path.join(DATA_DIR, "synthetic", f"{network}_N{n_samples}.npy"),
        "--ground-truth", os.path.join(DATA_DIR, "ground_truth", f"{network}_graph.pkl"),
        "--max-parents", str(MAX_PARENTS[network]),
        "--pop-size", str(POP_SIZE[network]),
        "--generations", str(GENERATIONS),
        "--seed", str(seed),
        "--output", output_dir,
        "--no-plot", "--no-params",
    ]

    try:
        subprocess.run(
            cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        return (network, prior, n_samples, seed, "ok")
    except subprocess.CalledProcessError as e:
        msg = e.stderr.decode().strip().split("\n")[-1] if e.stderr else str(e.returncode)
        return (network, prior, n_samples, seed, f"error: {msg}")


# ── 汇总 ─────────────────────────────────────────────────────────────

def generate_summary(networks, priors, sample_sizes, seeds):
    """读取所有运行结果，汇总到 summary.csv。

    每行 = 一次运行中 F1_skel 最高的 Pareto 解。
    """
    summary_path = os.path.join(OUT_DIR, "summary.csv")
    fieldnames = [
        "network", "prior", "n_samples", "seed",
        "n_pareto", "edges", "mdl", "sdiff",
        "shd", "f1", "shd_skel", "f1_skel", "runtime",
    ]

    rows = []
    missing = 0
    for network in networks:
        for prior in priors:
            for n_samples in sample_sizes:
                for seed in seeds:
                    run_dir = os.path.join(
                        OUT_DIR, f"{network}_{prior}_N{n_samples}", f"run_{seed}",
                    )
                    row = _extract_run_metrics(
                        network, prior, n_samples, seed, run_dir,
                    )
                    if row:
                        rows.append(row)
                    else:
                        missing += 1

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n汇总: {len(rows)} 行 → {summary_path}")
    if missing:
        print(f"  缺失: {missing} 个运行结果")


def _extract_run_metrics(network, prior, n_samples, seed, run_dir):
    """从单次运行提取最优解指标（按 F1_skel 选最佳 Pareto 解）。"""
    result_pkl = os.path.join(run_dir, "result.pkl")
    pareto_csv = os.path.join(run_dir, "pareto_front.csv")

    if not os.path.exists(result_pkl) or not os.path.exists(pareto_csv):
        return None

    # runtime 和 Pareto 解数
    try:
        with open(result_pkl, "rb") as f:
            result = pickle.load(f)
        runtime = round(result.runtime, 1)
        n_pareto = len(result.pareto_graphs)
    except Exception:
        return None

    # 从 pareto_front.csv 选 F1_skel 最高的解
    best = None
    best_f1 = -1.0
    try:
        with open(pareto_csv) as f:
            for row in csv.DictReader(f):
                f1_skel = float(row.get("f1_skel", 0))
                if f1_skel > best_f1:
                    best = row
                    best_f1 = f1_skel
    except Exception:
        return None

    if best is None:
        return None

    return {
        "network": network,
        "prior": prior,
        "n_samples": n_samples,
        "seed": seed,
        "n_pareto": n_pareto,
        "edges": best["edges"],
        "mdl": best["mdl"],
        "sdiff": best["sdiff"],
        "shd": best["shd"],
        "f1": best["f1"],
        "shd_skel": best["shd_skel"],
        "f1_skel": best["f1_skel"],
        "runtime": runtime,
    }


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = ArgumentParser(description="全量 MOEA/D 实验")
    parser.add_argument(
        "--networks", type=str, default=None,
        help=f"逗号分隔，默认: {','.join(NETWORKS)}",
    )
    parser.add_argument(
        "--priors", type=str, default=None,
        help=f"逗号分隔，默认: {','.join(PRIORS)}",
    )
    parser.add_argument(
        "--samples", type=str, default=None,
        help=f"逗号分隔，默认: {','.join(map(str, SAMPLE_SIZES))}",
    )
    parser.add_argument(
        "--seeds", type=str, default=None,
        help=f"逗号分隔，默认: 42..71 (30 seeds)",
    )
    parser.add_argument(
        "--workers", type=int, default=os.cpu_count(),
        help=f"并行 worker 数，默认: CPU 核心数 ({os.cpu_count()})",
    )
    parser.add_argument(
        "--summary-only", action="store_true",
        help="跳过运行，仅汇总已有结果",
    )
    args = parser.parse_args()

    networks = args.networks.split(",") if args.networks else NETWORKS
    priors = args.priors.split(",") if args.priors else PRIORS
    samples = [int(s) for s in args.samples.split(",")] if args.samples else SAMPLE_SIZES
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else SEEDS

    if args.summary_only:
        generate_summary(networks, priors, samples, seeds)
        return

    # ── 构建任务列表（跳过已完成） ────────────────────────
    tasks = []
    skipped = 0
    for net in networks:
        for prior in priors:
            for n in samples:
                for seed in seeds:
                    run_dir = os.path.join(
                        OUT_DIR, f"{net}_{prior}_N{n}", f"run_{seed}",
                    )
                    if os.path.exists(os.path.join(run_dir, "result.pkl")):
                        skipped += 1
                    else:
                        tasks.append((net, prior, n, seed))

    total = len(tasks) + skipped
    n_combos = len(networks) * len(priors) * len(samples)

    print("=" * 60)
    print("MOEA/D 全量实验")
    print(f"  矩阵: {len(networks)} 网络 × {len(priors)} 先验 × {len(samples)} 样本量 × {len(seeds)} 重复 = {total}")
    print(f"  组合: {n_combos} 个 (每个 {len(seeds)} 次重复)")
    print(f"  参数: gen={GENERATIONS}, pop={set(POP_SIZE[n] for n in networks)}")
    if skipped:
        print(f"  已完成: {skipped}, 待运行: {len(tasks)}")
    print(f"  workers: {args.workers}")
    print(f"  输出: {OUT_DIR}")
    print("=" * 60)

    if not tasks:
        print("\n全部已完成，直接生成汇总。")
        generate_summary(networks, priors, samples, seeds)
        return

    # ── 运行 ──────────────────────────────────────────────
    os.makedirs(OUT_DIR, exist_ok=True)
    done = skipped
    failed = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_single, *t): t for t in tasks}

        for future in as_completed(futures):
            net, prior, n, seed = futures[future]
            status = future.result()[4]
            done += 1

            if "error" in status:
                failed += 1
                print(f"  [{done}/{total}] FAIL {net}_{prior}_N{n} seed={seed}: {status}",
                      file=sys.stderr)
            elif done % 100 == 0 or done == total:
                elapsed = time.time() - t0
                rate = (done - skipped) / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                print(f"  [{done}/{total}] +{done - skipped} 完成, "
                      f"{failed} 失败, {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

    elapsed = time.time() - t0
    print(f"\n运行完成: {done - skipped} 新完成 + {skipped} 已有 = {done}/{total}"
          f" ({failed} 失败, {elapsed:.0f}s)")

    # ── 汇总 ──────────────────────────────────────────────
    generate_summary(networks, priors, samples, seeds)


if __name__ == "__main__":
    main()
