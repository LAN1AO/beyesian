"""精细化 profile: 在 crossover 内部测量各阶段耗时。"""
from __future__ import annotations
import time, random, json, argparse
from collections import defaultdict
import numpy as np
from src.config import MOEADConfig
from src.prior import PriorNetwork
from src.moead import MOEAD
from src.operators import crossover as _orig_crossover
from src.graph import DirectedGraph


# Per-call timing storage
_call_timings = []


def profiled_crossover(*args, **kwargs):
    """返回 child, 同时记录内部各阶段耗时到 _call_timings"""
    t = {}
    t0 = time.perf_counter()

    parent1 = args[0]
    parent2 = args[1]
    mdl_score = args[2]
    sdiff_score = args[3]
    weight = args[4]
    ideal = args[5]
    nadir = args[6]
    eps = args[7] if len(args) > 7 else 1e-8
    rng = kwargs.get("rng")
    parent1_scores = kwargs.get("parent1_scores")
    parent2_scores = kwargs.get("parent2_scores")
    score_cache = kwargs.get("score_cache")
    crossover_type = kwargs.get("crossover_type", "sequential")

    if rng is None:
        rng = random.Random()
    n_nodes = parent1.n_nodes
    max_parents = parent1.max_parents
    child = DirectedGraph(n_nodes, max_parents=max_parents)
    check_cycles = (crossover_type != "no-cycle-check")
    sort_by_diff = (crossover_type == "score-diff-sort")

    # Stage timers
    t_gp = t_score = t_cheb = t_edge = t_pre_gp = t_pre_score = t_pre_cheb = 0.0

    range_ = np.maximum(nadir - ideal, eps)

    node_order = list(range(n_nodes))
    if sort_by_diff:
        node_diffs = []
        for node in range(n_nodes):
            t1 = time.perf_counter()
            p1p = parent1.get_parents(node)
            p2p = parent2.get_parents(node)
            t_pre_gp += time.perf_counter() - t1

            if set(p1p) == set(p2p):
                node_diffs.append((node, 0.0))
                continue

            t1 = time.perf_counter()
            if parent1_scores is not None and node in parent1_scores:
                mdl1, sd1 = parent1_scores[node]
            else:
                key = (node, frozenset(p1p))
                cached = score_cache.get(key) if score_cache else None
                if cached:
                    mdl1, sd1 = cached
                else:
                    mdl1 = mdl_score.score_node(node, p1p)
                    sd1 = sdiff_score.score_node(node, p1p)
                    if score_cache is not None:
                        score_cache[key] = (mdl1, sd1)
            if parent2_scores is not None and node in parent2_scores:
                mdl2, sd2 = parent2_scores[node]
            else:
                key = (node, frozenset(p2p))
                cached = score_cache.get(key) if score_cache else None
                if cached:
                    mdl2, sd2 = cached
                else:
                    mdl2 = mdl_score.score_node(node, p2p)
                    sd2 = sdiff_score.score_node(node, p2p)
                    if score_cache is not None:
                        score_cache[key] = (mdl2, sd2)
            t_pre_score += time.perf_counter() - t1

            t1 = time.perf_counter()
            f1 = np.array([mdl1, sd1])
            f2 = np.array([mdl2, sd2])
            g1 = np.max(weight * np.abs(f1 - ideal) / range_)
            g2 = np.max(weight * np.abs(f2 - ideal) / range_)
            node_diffs.append((node, abs(g1 - g2)))
            t_pre_cheb += time.perf_counter() - t1

        node_diffs.sort(key=lambda x: x[1], reverse=True)
        node_order = [n for n, _ in node_diffs]
    else:
        rng.shuffle(node_order)

    # Count nodes where parents differ (in main loop)
    n_diff = 0

    # Main loop
    for node in node_order:
        t1 = time.perf_counter()
        p1_parents = parent1.get_parents(node)
        p2_parents = parent2.get_parents(node)
        t_gp += time.perf_counter() - t1

        if set(p1_parents) == set(p2_parents):
            selected = p1_parents
        else:
            n_diff += 1
            t1 = time.perf_counter()
            if parent1_scores is not None and node in parent1_scores:
                mdl1, sd1 = parent1_scores[node]
            else:
                key = (node, frozenset(p1_parents))
                cached = score_cache.get(key) if score_cache else None
                if cached:
                    mdl1, sd1 = cached
                else:
                    mdl1 = mdl_score.score_node(node, p1_parents)
                    sd1 = sdiff_score.score_node(node, p1_parents)
                    if score_cache is not None:
                        score_cache[key] = (mdl1, sd1)
            if parent2_scores is not None and node in parent2_scores:
                mdl2, sd2 = parent2_scores[node]
            else:
                key = (node, frozenset(p2_parents))
                cached = score_cache.get(key) if score_cache else None
                if cached:
                    mdl2, sd2 = cached
                else:
                    mdl2 = mdl_score.score_node(node, p2_parents)
                    sd2 = sdiff_score.score_node(node, p2_parents)
                    if score_cache is not None:
                        score_cache[key] = (mdl2, sd2)
            t_score += time.perf_counter() - t1

            t1 = time.perf_counter()
            f1 = np.array([mdl1, sd1])
            f2 = np.array([mdl2, sd2])
            g1 = np.max(weight * np.abs(f1 - ideal) / range_)
            g2 = np.max(weight * np.abs(f2 - ideal) / range_)
            selected = p1_parents if g1 <= g2 else p2_parents
            t_cheb += time.perf_counter() - t1

        t1 = time.perf_counter()
        for p in selected:
            if check_cycles:
                child.add_edge(p, node)
            else:
                if p != node and not child.adj[p, node]:
                    if max_parents is None or child.get_in_degree(node) < max_parents:
                        child.adj[p, node] = 1
        t_edge += time.perf_counter() - t1

    if not check_cycles:
        from src.operators import _fix_cycles
        _fix_cycles(child, rng)

    t_total = time.perf_counter() - t0
    _call_timings.append({
        "total": t_total,
        "gp": t_gp, "score": t_score, "cheb": t_cheb, "edge": t_edge,
        "pre_gp": t_pre_gp, "pre_score": t_pre_score, "pre_cheb": t_pre_cheb,
        "n_diff": n_diff, "n_nodes": n_nodes,
    })
    return child


def print_summary(label, timings):
    """Print summary statistics for collected timings."""
    if not timings:
        return
    print(f"\n{label}:")
    keys = ["total", "gp", "score", "cheb", "edge", "pre_gp", "pre_score", "pre_cheb"]
    names = {"total": "总耗时", "gp": "get_parents", "score": "score_lookup",
             "cheb": "chebyshev", "edge": "add_edge",
             "pre_gp": "预计算-gp", "pre_score": "预计算-score", "pre_cheb": "预计算-cheb"}
    total_ms = np.mean([x["total"] for x in timings]) * 1000
    print(f"  {'Stage':<16} {'Mean(ms)':>10} {'%':>6}")
    print(f"  {'-'*16} {'-'*10} {'-'*6}")
    for k in keys:
        vals = [x[k] for x in timings if x.get(k, 0) > 0 or k == "total"]
        if not vals or sum(vals) == 0:
            continue
        mean_ms = np.mean(vals) * 1000
        pct = mean_ms / total_ms * 100 if total_ms > 0 else 0
        print(f"  {names.get(k, k):<16} {mean_ms:>8.3f}ms {pct:>5.1f}%")
    nd = [x.get("n_diff", 0) for x in timings]
    print(f"  avg n_diff: {np.mean(nd):.0f}/{timings[0].get('n_nodes', '?')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crossover-type", type=str, default="sequential")
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading andes...")
    prior_graph, node_names, n_states = PriorNetwork.from_pgmpy_model("andes")
    prior_graph = PriorNetwork.perturb(prior_graph, n_changes=6, seed=args.seed)
    data, _, _ = PriorNetwork.generate_data("andes", n_samples=10000, seed=args.seed)

    config = MOEADConfig(
        n_nodes=len(node_names), n_states=n_states,
        max_parents=7,
        crossover_type=args.crossover_type,
        max_symmetric_diff=len(node_names) * 7 + len(prior_graph.get_edges()),
        n_weight_vectors=300, n_neighbors=30,
        n_generations=args.generations,
        prob_neighbor_mating=0.9, max_replacements=2,
        mutation_prob=0.3, mutation_ops_min=2, mutation_ops_max=6,
        data=data, random_seed=args.seed,
    )

    moead = MOEAD(config, prior_graph, data, node_names)

    # Patch crossover in BOTH modules:
    # - src.operators.crossover (where it's defined)
    # - src.moead.crossover (where `from src.operators import crossover` creates a local name)
    import src.operators as op_mod
    import src.moead as moead_mod
    op_mod.crossover = profiled_crossover
    moead_mod.crossover = profiled_crossover
    # Verify patch took effect
    assert moead_mod.crossover is profiled_crossover, f"Patch failed: {moead_mod.crossover}"

    global _call_timings
    _call_timings = []

    print(f"Running ({args.crossover_type}, {args.generations} gens)...")
    t0 = time.time()
    result = moead.run()
    print(f"\nTotal: {time.time() - t0:.1f}s, {len(_call_timings)} crossover calls")

    # Split into early (first 20%), mid, late (last 30%)
    n = len(_call_timings)
    early = _call_timings[:max(1, n // 5)]
    mid = _call_timings[n // 3: 2 * n // 3]
    late = _call_timings[2 * n // 3:]

    print_summary(f"早期 ({len(early)} calls)", early)
    print_summary(f"中期 ({len(mid)} calls)", mid)
    print_summary(f"后期 ({len(late)} calls)", late)

    out = {
        "crossover_type": args.crossover_type,
        "early": early,
        "mid": mid,
        "late": late,
    }
    path = f"/tmp/profile_v2_{args.crossover_type}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
