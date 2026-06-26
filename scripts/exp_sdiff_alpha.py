#!/usr/bin/env python3
"""sdiff_alpha 先验置信度对照实验。

矩阵: alarm × 5 先验(gt/mild/moderate/severe/random) × α{1.0,0.5,0.25,0.1} × 3 seed = 60 次。
验证 α(sdiff 项缩放)作为"先验置信度旋钮": α↓ 削弱先验牵引。
预测: gt 先验 F1 随 α↓ 下降; random/severe 随 α↓ 上升(数据压过坏先验)。

用法:
    python scripts/exp_sdiff_alpha.py                  # 全量(断点续跑)
    python scripts/exp_sdiff_alpha.py --summary-only   # 仅汇总已有结果
    python scripts/exp_sdiff_alpha.py --output DIR      # 自定义输出目录
"""

import csv
import os
import subprocess
import sys
import time
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN_PY = os.path.join(ROOT, "main.py")
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "output", "exp_sdiff_alpha")

NETWORK = "alarm"
PRIORS = ["gt", "mild", "moderate", "severe", "random"]
ALPHAS = [1.0, 0.5, 0.25, 0.1]
SEEDS = [42, 43, 44]
N_SAMPLES = 1000
POP_SIZE = 100
MAX_PARENTS = 6
GENERATIONS = 3000


def run_single(prior, alpha, seed, out_dir):
    """跑一次 MOEA/D。断点续跑: result.pkl 存在则跳过。"""
    output_dir = os.path.join(out_dir, f"{NETWORK}_{prior}_a{alpha}", f"run_{seed}")
    if os.path.exists(os.path.join(output_dir, "result.pkl")):
        return (prior, alpha, seed, "skipped")
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        sys.executable, MAIN_PY,
        "--prior-file", os.path.join(DATA_DIR, "priors", f"{NETWORK}_{prior}.pkl"),
        "--data-file", os.path.join(DATA_DIR, "synthetic", f"{NETWORK}_N{N_SAMPLES}.npy"),
        "--ground-truth", os.path.join(DATA_DIR, "ground_truth", f"{NETWORK}_graph.pkl"),
        "--max-parents", str(MAX_PARENTS),
        "--pop-size", str(POP_SIZE),
        "--generations", str(GENERATIONS),
        "--sdiff-alpha", str(alpha),
        "--seed", str(seed),
        "--output", output_dir,
        "--no-plot", "--no-params",
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE)
        return (prior, alpha, seed, "ok")
    except subprocess.CalledProcessError as e:
        msg = e.stderr.decode().strip().split("\n")[-1] if e.stderr else str(e.returncode)
        return (prior, alpha, seed, f"error: {msg}")


def _best_row(pareto_csv):
    """取 pareto_front.csv 中 F1_skel 最高的行(同 run_experiments 规则)。"""
    best, best_f1 = None, -1.0
    try:
        with open(pareto_csv) as f:
            for row in csv.DictReader(f):
                f1 = float(row.get("f1_skel", 0))
                if f1 > best_f1:
                    best, best_f1 = row, f1
    except FileNotFoundError:
        return None
    return best


def generate_summary(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for prior in PRIORS:
        for alpha in ALPHAS:
            for seed in SEEDS:
                run_dir = os.path.join(out_dir, f"{NETWORK}_{prior}_a{alpha}",
                                       f"run_{seed}")
                best = _best_row(os.path.join(run_dir, "pareto_front.csv"))
                if best is None:
                    continue
                rows.append({
                    "prior": prior, "alpha": alpha, "seed": seed,
                    "f1_skel": best["f1_skel"], "shd_skel": best["shd_skel"],
                    "sdiff": best["sdiff"],
                })
    summary_path = os.path.join(out_dir, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["prior", "alpha", "seed",
                                          "f1_skel", "shd_skel", "sdiff"])
        w.writeheader()
        w.writerows(rows)
    print(f"汇总: {len(rows)} 行 → {summary_path}")
    if rows:
        _plot(rows, out_dir)


def _plot(rows, out_dir):
    """F1_skel / SHD_skel vs α，每先验一条线(3 seed mean±std)。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for prior in PRIORS:
        f1_mean, f1_std, shd_mean = [], [], []
        for alpha in ALPHAS:
            vals = [r for r in rows if r["prior"] == prior and r["alpha"] == alpha]
            f1 = [float(r["f1_skel"]) for r in vals]
            shd = [float(r["shd_skel"]) for r in vals]
            f1_mean.append(np.mean(f1) if f1 else np.nan)
            f1_std.append(np.std(f1) if f1 else 0)
            shd_mean.append(np.mean(shd) if shd else np.nan)
        ax1.errorbar(ALPHAS, f1_mean, yerr=f1_std, marker="o", capsize=3, label=prior)
        ax2.plot(ALPHAS, shd_mean, marker="s", label=prior)
    for ax, ylab, title in [(ax1, "best F1_skel", "F1_skel vs alpha"),
                            (ax2, "best SHD_skel", "SHD_skel vs alpha")]:
        ax.set_xlabel("sdiff_alpha (smaller = weaker prior pull)")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.set_xscale("log")
        ax.invert_xaxis()  # α 从大(1.0)到小(0.1)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle(f"{NETWORK}: sdiff_alpha as prior-confidence knob "
                 f"(N{N_SAMPLES}, {len(SEEDS)} seeds)", fontweight="bold")
    fig.tight_layout()
    path = os.path.join(out_dir, "f1_vs_alpha.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"图表: {path}")


def main():
    p = ArgumentParser(description="sdiff_alpha 先验置信度对照实验")
    p.add_argument("--output", default=None, help="输出目录")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--summary-only", action="store_true", help="仅汇总")
    args = p.parse_args()
    out_dir = args.output or OUT_DIR

    if args.summary_only:
        generate_summary(out_dir)
        return

    tasks = [(prior, alpha, seed)
             for prior in PRIORS for alpha in ALPHAS for seed in SEEDS]
    os.makedirs(out_dir, exist_ok=True)
    print(f"实验: {NETWORK} × {len(PRIORS)}先验 × {len(ALPHAS)}α × {len(SEEDS)}seed "
          f"= {len(tasks)} 次")
    print(f"  gen={GENERATIONS}, pop={POP_SIZE}, workers={args.workers}, 输出={out_dir}")

    done = failed = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_single, *t, out_dir): t for t in tasks}
        for fut in as_completed(futs):
            prior, alpha, seed, status = fut.result()
            done += 1
            if "error" in status:
                failed += 1
                print(f"  [{done}/{len(tasks)}] FAIL {prior} a{alpha} s{seed}: {status}",
                      file=sys.stderr)
            elif done % 10 == 0 or done == len(tasks):
                print(f"  [{done}/{len(tasks)}] {failed} 失败, "
                      f"{time.time()-t0:.0f}s")
    print(f"完成: {done-failed}/{len(tasks)} ok, {failed} 失败, {time.time()-t0:.0f}s")
    generate_summary(out_dir)


if __name__ == "__main__":
    main()
