"""并行多次运行 MOEA/D 结构学习，汇总 Pareto 前沿到一张图中。

用法:
  python3 scripts/batch_run.py                       # 使用默认参数
  python3 scripts/batch_run.py --runs 30 --workers 8  # 自定义运行次数和并行度
"""

from __future__ import annotations

import argparse
import os
import pickle
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

# 确保项目根目录在 sys.path 中（解决 pickle 和 import 路径问题）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_PROJECT_ROOT)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ── 默认参数 ──────────────────────────────────────────────

DEFAULTS = {
    "model": "alarm",
    "pop_size": 100,
    "generations": 10000,
    "max_sdiff": 50,
    "n_samples": 10000,
    "max_parents": 4,
}


def run_one(seed: int, base_output: str) -> str:
    """执行一次 MOEA/D 运行，返回输出目录路径。"""
    output_dir = os.path.join(base_output, f"batch_{seed}")
    cmd = [
        sys.executable, "main.py",
        "--model", DEFAULTS["model"],
        "--pop-size", str(DEFAULTS["pop_size"]),
        "--generations", str(DEFAULTS["generations"]),
        "--max-sdiff", str(DEFAULTS["max_sdiff"]),
        "--n-samples", str(DEFAULTS["n_samples"]),
        "--max-parents", str(DEFAULTS["max_parents"]),
        "--seed", str(seed),
        "--output", output_dir,
        "--no-plot",
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return output_dir


def collect_results(output_dirs: list[str]) -> list[dict]:
    """从各输出目录收集 Pareto 前沿数据。"""
    results = []
    for d in output_dirs:
        pkl_path = os.path.join(d, "result.pkl")
        if not os.path.exists(pkl_path):
            print(f"  [warn] 缺少 result.pkl: {pkl_path}")
            continue
        with open(pkl_path, "rb") as f:
            result = pickle.load(f)
        name = os.path.basename(d)
        results.append({
            "name": name,
            "pareto_f": result.pareto_f.copy(),
            "runtime": result.runtime,
        })
        print(f"  {name}: {len(result.pareto_f)} 个 Pareto 解, "
              f"耗时 {result.runtime:.0f}s")
    return results


def non_dominated_mask(f_values: np.ndarray) -> np.ndarray:
    """返回非支配解的布尔掩码（最小化两个目标）。"""
    n = len(f_values)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        for j in range(n):
            if i == j or not mask[j]:
                continue
            if (f_values[j, 0] <= f_values[i, 0] and f_values[j, 1] <= f_values[i, 1]
                    and (f_values[j, 0] < f_values[i, 0] or f_values[j, 1] < f_values[i, 1])):
                mask[i] = False
                break
    return mask


def plot_combined(results: list[dict], save_path: str) -> plt.Figure:
    """将所有运行的 Pareto 前沿绘制到一张图中。"""
    fig, ax = plt.subplots(figsize=(10, 8))

    n_results = len(results)
    if n_results <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, n_results))
    else:
        colors = plt.cm.tab20(np.arange(n_results) % 20 / 20.0)

    # 各次运行的 Pareto 前沿（半透明）
    for i, r in enumerate(results):
        f = r["pareto_f"]
        ax.scatter(
            f[:, 0], f[:, 1],
            c=[colors[i % 20]], s=25, alpha=0.4,
            edgecolors="none",
            label=r["name"],
        )

    # 汇总所有解，计算全局非支配前沿
    all_f = np.vstack([r["pareto_f"] for r in results])
    global_mask = non_dominated_mask(all_f)
    global_pf = all_f[global_mask]
    order = np.argsort(global_pf[:, 0])
    global_pf = global_pf[order]

    ax.plot(
        global_pf[:, 0], global_pf[:, 1],
        "k-", linewidth=2, alpha=0.7, label="Global Pareto Front",
    )
    ax.scatter(
        global_pf[:, 0], global_pf[:, 1],
        c="black", s=50, marker="o", alpha=0.9,
        edgecolors="white", linewidth=0.5, zorder=5,
    )

    # 先验网络位置 (Sdiff=0)
    zero_sdiff = global_pf[global_pf[:, 1] == 0]
    if len(zero_sdiff) > 0:
        ax.scatter(
            [zero_sdiff[0, 0]], [0],
            c="red", s=150, marker="*",
            edgecolors="darkred", linewidth=1,
            label="Prior Network", zorder=10,
        )

    ax.set_xlabel("MDL Score", fontsize=13)
    ax.set_ylabel("Structural Symmetric Difference", fontsize=13)
    ax.set_title(
        f"Pareto Fronts — {n_results} runs ({DEFAULTS['model']}, "
        f"pop={DEFAULTS['pop_size']}, gen={DEFAULTS['generations']})",
        fontsize=14,
    )
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"汇总图已保存: {save_path}")
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="并行多次运行 MOEA/D，汇总 Pareto 前沿"
    )
    parser.add_argument("--runs", type=int, default=20,
                        help="运行次数 (默认: 20)")
    parser.add_argument("--workers", type=int, default=None,
                        help="并行 worker 数 (默认: CPU 核心数)")
    parser.add_argument("--output", type=str, default="./output/batch",
                        help="总输出目录 (默认: ./output/batch)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    seeds = list(range(42, 42 + args.runs))

    print(f"项目目录: {_PROJECT_ROOT}")
    print(f"启动 {args.runs} 次并行运行 (workers={args.workers or 'auto'})...")
    print(f"参数: model={DEFAULTS['model']}, "
          f"pop={DEFAULTS['pop_size']}, gen={DEFAULTS['generations']}, "
          f"samples={DEFAULTS['n_samples']}, max_sdiff={DEFAULTS['max_sdiff']}, "
          f"max_parents={DEFAULTS['max_parents']}")
    print(f"输出目录: {args.output}/")

    # ── 并行执行 ──────────────────────────────────────────
    output_dirs = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_one, seed, args.output): seed
            for seed in seeds
        }
        for future in as_completed(futures):
            seed = futures[future]
            try:
                d = future.result()
                output_dirs.append(d)
                print(f"  [ok] seed={seed} ({len(output_dirs)}/{args.runs})")
            except subprocess.CalledProcessError as e:
                print(f"  [fail] seed={seed}: 返回码 {e.returncode}")

    print(f"\n完成 {len(output_dirs)}/{args.runs} 次运行")

    # ── 收集结果 ──────────────────────────────────────────
    print("收集结果...")
    results = collect_results(output_dirs)

    if not results:
        print("没有可用的结果，退出。")
        return

    # ── 汇总统计 ──────────────────────────────────────────
    runtimes = [r["runtime"] for r in results if r["runtime"] > 0]
    pareto_sizes = [len(r["pareto_f"]) for r in results]
    print(f"\n统计 ({len(results)} 次运行):")
    if runtimes:
        print(f"  耗时: 均值 {np.mean(runtimes):.0f}s, "
              f"最小 {np.min(runtimes):.0f}s, 最大 {np.max(runtimes):.0f}s")
    print(f"  Pareto 解数: 均值 {np.mean(pareto_sizes):.1f}, "
          f"最小 {np.min(pareto_sizes)}, 最大 {np.max(pareto_sizes)}")

    # ── 保存全局 Pareto 前沿 CSV ─────────────────────────
    all_f = np.vstack([r["pareto_f"] for r in results])
    global_mask = non_dominated_mask(all_f)
    global_pf = all_f[global_mask]
    order = np.argsort(global_pf[:, 0])
    global_pf = global_pf[order]
    csv_path = os.path.join(args.output, "global_pareto.csv")
    np.savetxt(csv_path, global_pf, delimiter=",",
               header="mdl,sdiff", comments="", fmt="%.4f,%.0f")
    print(f"全局 Pareto 前沿 CSV: {csv_path} ({len(global_pf)} 个解)")

    # ── 绘图 ──────────────────────────────────────────────
    plot_path = os.path.join(args.output, "combined_pareto.png")
    plot_combined(results, plot_path)

    print("全部完成。")


if __name__ == "__main__":
    main()
