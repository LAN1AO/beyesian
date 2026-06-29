#!/usr/bin/env python3
"""通用 MOEA/D 实验台（配置文件驱动）。

对一个超参矩阵逐 cell 调 main.py 跑 MOEA/D、断点续跑、汇总 best-F1_skel 行成
CSV。所有实验维度与超参均来自 JSON 配置文件，命令行不接收任何实验参数。

用法:
    python scripts/run_experiment.py <config.json>                # 全量(断点续跑)+ 汇总
    python scripts/run_experiment.py <config.json> --summary-only # 仅汇总已有结果

配置文件字段 (JSON):
    output       str          输出目录(必填)
    workers      int          并行 worker 数(默认 CPU 核心数)
    networks     [str]        网络名列表    ┐
    priors       [str]        先验档列表    │ 五维笛卡尔积
    alphas       [float]      sdiff_alpha   ├─ = 全部实验 cell
    n_samples    [int]        样本量列表    │  (alphas 用浮点写法, 如 1.0/0.05,
    seeds        [int]        随机种子列表  ┘   以与目录名/续跑保持一致)
    params       {str: val}   透传 main.py 的全局超参默认(键见 PARAM_FLAGS/BOOL_FLAGS)
    per_network  {net: {...}} 可选, 按网络覆盖部分 params(规模不同的网络用不同 pop/parents)

数据假设已由 scripts/prepare_data.py 预生成:
    data/priors/{net}_{prior}.pkl  data/synthetic/{net}_N{n}.npy  data/ground_truth/{net}_graph.pkl

输出目录: {output}/{net}_{prior}_a{alpha}_N{n}/run_{seed}/  (result.pkl 存在则断点续跑跳过)
"""

import csv
import json
import os
import subprocess
import sys
import time
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN_PY = os.path.join(ROOT, "main.py")
DATA_DIR = os.path.join(ROOT, "data")

# params 键 → main.py 命令行 flag。带值: None 跳过; 布尔: 仅 True 时追加 flag。
PARAM_FLAGS = {
    "pop_size": "--pop-size",
    "max_parents": "--max-parents",
    "generations": "--generations",
    "neighbors": "--neighbors",
    "prob_neighbor": "--prob-neighbor",
    "max_replace": "--max-replace",
    "mutation_prob": "--mutation-prob",
    "mutation_ops_min": "--mutation-ops-min",
    "mutation_ops_max": "--mutation-ops-max",
    "crossover_type": "--crossover-type",
    "mdl_penalty": "--mdl-penalty",
    "max_sdiff": "--max-sdiff",
}
BOOL_FLAGS = {"track_sdiff": "--track-sdiff"}

SUMMARY_FIELDS = ["network", "prior", "alpha", "n_samples", "seed", "n_pareto",
                  "edges", "mdl", "sdiff", "shd", "f1", "shd_skel", "f1_skel"]


def _cells(cfg):
    return list(product(cfg["networks"], cfg["priors"], cfg["alphas"],
                        cfg["n_samples"], cfg["seeds"]))


def _run_dir(cfg, net, prior, alpha, n, seed):
    return os.path.join(cfg["output"], f"{net}_{prior}_a{alpha}_N{n}",
                        f"run_{seed}")


def _params_for(cfg, net):
    """全局 params 叠加该网络的 per_network 覆盖。"""
    p = dict(cfg.get("params", {}))
    p.update(cfg.get("per_network", {}).get(net, {}))
    return p


def run_single(cfg, net, prior, alpha, n, seed):
    """跑一次 MOEA/D。断点续跑: result.pkl 存在则跳过。"""
    out = _run_dir(cfg, net, prior, alpha, n, seed)
    if os.path.exists(os.path.join(out, "result.pkl")):
        return (net, prior, alpha, n, seed, "skipped")
    os.makedirs(out, exist_ok=True)
    cmd = [
        sys.executable, MAIN_PY,
        "--prior-file", os.path.join(DATA_DIR, "priors", f"{net}_{prior}.pkl"),
        "--data-file", os.path.join(DATA_DIR, "synthetic", f"{net}_N{n}.npy"),
        "--ground-truth", os.path.join(DATA_DIR, "ground_truth", f"{net}_graph.pkl"),
        "--sdiff-alpha", str(alpha),
        "--seed", str(seed),
        "--output", out,
        "--no-plot", "--no-params",
    ]
    params = _params_for(cfg, net)
    for key, flag in PARAM_FLAGS.items():
        val = params.get(key)
        if val is not None:
            cmd += [flag, str(val)]
    for key, flag in BOOL_FLAGS.items():
        if params.get(key):
            cmd.append(flag)
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE)
        return (net, prior, alpha, n, seed, "ok")
    except subprocess.CalledProcessError as e:
        msg = e.stderr.decode().strip().split("\n")[-1] if e.stderr else str(e.returncode)
        return (net, prior, alpha, n, seed, f"error: {msg}")


def _best_row(pareto_csv):
    """取 pareto_front.csv 中 f1_skel 最高的行, 同时返回前沿解数 n_pareto。"""
    best, best_f1, n = None, -1.0, 0
    try:
        with open(pareto_csv) as f:
            for row in csv.DictReader(f):
                n += 1
                f1 = float(row.get("f1_skel", 0))
                if f1 > best_f1:
                    best, best_f1 = row, f1
    except FileNotFoundError:
        return None, 0
    return best, n


def generate_summary(cfg):
    rows = []
    for net, prior, alpha, n, seed in _cells(cfg):
        out = _run_dir(cfg, net, prior, alpha, n, seed)
        best, n_pareto = _best_row(os.path.join(out, "pareto_front.csv"))
        if best is None:
            continue
        rows.append({
            "network": net, "prior": prior, "alpha": alpha,
            "n_samples": n, "seed": seed, "n_pareto": n_pareto,
            "edges": best["edges"], "mdl": best["mdl"], "sdiff": best["sdiff"],
            "shd": best["shd"], "f1": best["f1"],
            "shd_skel": best["shd_skel"], "f1_skel": best["f1_skel"],
        })
    os.makedirs(cfg["output"], exist_ok=True)
    path = os.path.join(cfg["output"], "summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"汇总: {len(rows)} 行 → {path}")


def main():
    p = ArgumentParser(description="通用 MOEA/D 实验台(配置文件驱动)")
    p.add_argument("config", help="实验配置文件 (JSON)")
    p.add_argument("--summary-only", action="store_true", help="仅汇总已有结果")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    cfg.setdefault("workers", os.cpu_count())

    if args.summary_only:
        generate_summary(cfg)
        return

    cells = _cells(cfg)
    os.makedirs(cfg["output"], exist_ok=True)
    print(f"实验: {len(cfg['networks'])}网络 × {len(cfg['priors'])}先验 × "
          f"{len(cfg['alphas'])}α × {len(cfg['n_samples'])}N × "
          f"{len(cfg['seeds'])}seed = {len(cells)} 次")
    print(f"  workers={cfg['workers']}, 输出={cfg['output']}")

    done = failed = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=cfg["workers"]) as ex:
        futs = {ex.submit(run_single, cfg, *c): c for c in cells}
        for fut in as_completed(futs):
            net, prior, alpha, n, seed, status = fut.result()
            done += 1
            if "error" in status:
                failed += 1
                print(f"  [{done}/{len(cells)}] FAIL {net} {prior} a{alpha} "
                      f"N{n} s{seed}: {status}", file=sys.stderr)
            elif done % 10 == 0 or done == len(cells):
                print(f"  [{done}/{len(cells)}] {failed} 失败, {time.time()-t0:.0f}s")
    print(f"完成: {done-failed}/{len(cells)} ok, {failed} 失败, {time.time()-t0:.0f}s")
    generate_summary(cfg)


if __name__ == "__main__":
    main()
