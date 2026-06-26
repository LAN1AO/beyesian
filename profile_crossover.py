"""Profile crossover 算子在 andes 上的耗时分布。"""
from __future__ import annotations

import os, sys, time, random, pickle, json, argparse
from collections import defaultdict

import numpy as np

from src.config import MOEADConfig
from src.prior import PriorNetwork
from src.moead import MOEAD
from src.operators import crossover as crossover_fn

# Monkey-patch MOEAD.run to collect per-step timing
_original_run = MOEAD.run


def profiled_run(self):
    config = self.config
    rng = random.Random(config.random_seed)
    history = []
    t0 = time.time()

    # Timing accumulators (per generation)
    timing = defaultdict(list)

    for gen in range(config.n_generations):
        gen_t0 = time.time()

        # Per-step timing within generation
        step_cross = 0.0
        step_mutate = 0.0
        step_eval = 0.0
        step_neighbor = 0.0
        step_other = 0.0

        nadir_comp_start = time.time()
        from src.decomposition import compute_nadir
        nadir = compute_nadir(self.F)
        nadir_time = time.time() - nadir_comp_start

        for i in range(self.pop_size):
            # 1. Select parents
            t1 = time.time()
            from src.operators import select_parents
            k, l = select_parents(
                self.pop_size, i, self.neighbors,
                config.prob_neighbor_mating, rng,
            )
            step_other += time.time() - t1

            # 2. Crossover
            t1 = time.time()
            child = crossover_fn(
                self.population[k],
                self.population[l],
                self.mdl_score,
                self.sdiff_score,
                self.weights[i],
                self.ideal,
                nadir,
                config.eps,
                rng,
                parent1_scores=self.node_scores[k],
                parent2_scores=self.node_scores[l],
                score_cache=self._score_cache,
                crossover_type=config.crossover_type,
            )
            step_cross += time.time() - t1

            # 3. Mutation
            t1 = time.time()
            if rng.random() < config.mutation_prob:
                from src.operators import mutate
                child = mutate(child, config, rng)
                step_mutate += time.time() - t1
            else:
                step_mutate += time.time() - t1

            # 4. Evaluate
            t1 = time.time()
            child_f, child_node_scores = self._evaluate(child)
            step_eval += time.time() - t1

            # 5. Check constraint
            if child_f[1] > config.max_symmetric_diff:
                continue

            # 6. Update ideal
            t1 = time.time()
            from src.decomposition import update_ideal
            self.ideal = update_ideal(child_f, self.ideal)
            step_other += time.time() - t1

            # 7. Update neighbors
            t1 = time.time()
            from src.decomposition import chebyshev_aggregate
            neighbor_list = self.neighbors[i].copy()
            rng.shuffle(neighbor_list)
            n_replaced = 0
            for j in neighbor_list:
                if n_replaced >= config.max_replacements:
                    break
                g_child = chebyshev_aggregate(
                    child_f, self.weights[j], self.ideal, nadir, config.eps
                )
                g_curr = chebyshev_aggregate(
                    self.F[j], self.weights[j], self.ideal, nadir, config.eps
                )
                if g_child <= g_curr:
                    self.population[j] = child.copy()
                    self.F[j] = child_f.copy()
                    self.node_scores[j] = child_node_scores.copy()
                    n_replaced += 1
            step_neighbor += time.time() - t1

        gen_total = time.time() - gen_t0

        timing["cross"].append(step_cross)
        timing["mutate"].append(step_mutate)
        timing["eval"].append(step_eval)
        timing["neighbor"].append(step_neighbor)
        timing["other"].append(step_other)
        timing["nadir"].append(nadir_time)
        timing["total"].append(gen_total)

        # Record Pareto history
        from src.decomposition import non_dominated_sort
        pareto_mask = non_dominated_sort(self.F)
        history.append(self.F[pareto_mask].copy())

        pct = (gen + 1) / config.n_generations
        bar_len = 30
        filled = int(bar_len * pct)
        bar = "█" * filled + "░" * (bar_len - filled)
        # Print per-step breakdown for latest gen
        cross_pct = step_cross / gen_total * 100
        eval_pct = step_eval / gen_total * 100
        mut_pct = step_mutate / gen_total * 100
        neigh_pct = step_neighbor / gen_total * 100
        other_pct = step_other / gen_total * 100
        print(f"\r  [{bar}] {gen+1}/{config.n_generations} "
              f"({gen_total:.1f}s C:{cross_pct:.0f}% E:{eval_pct:.0f}% "
              f"M:{mut_pct:.0f}% N:{neigh_pct:.0f}% O:{other_pct:.0f}%)",
              end="", flush=True)

    print()
    runtime = time.time() - t0

    # Build result
    from src.decomposition import non_dominated_sort
    pareto_mask = non_dominated_sort(self.F)
    pareto_indices = np.where(pareto_mask)[0]

    from src.moead import MOEADResult
    result = MOEADResult(
        pareto_graphs=[self.population[i] for i in pareto_indices],
        pareto_f=self.F[pareto_indices],
        population=self.population,
        population_f=self.F,
        history=history,
        sdiff_history=[],
        ideal=self.ideal,
        config=config,
        node_names=self.node_names,
        runtime=runtime,
    )
    result._timing = dict(timing)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crossover-type", type=str, default="sequential",
                        choices=["sequential", "score-diff-sort"])
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"加载 andes 网络...")
    prior_graph, node_names, n_states = PriorNetwork.from_pgmpy_model("andes")
    prior_graph = PriorNetwork.perturb(prior_graph, n_changes=6, seed=args.seed)
    print(f"  节点: {len(node_names)}, 先验边: {len(prior_graph.get_edges())}")

    print(f"生成数据...")
    data, _, _ = PriorNetwork.generate_data("andes", n_samples=10000, seed=args.seed)
    print(f"  样本: {data.shape}")

    config = MOEADConfig(
        n_nodes=len(node_names),
        n_states=n_states,
        max_parents=7,
        crossover_type=args.crossover_type,
        max_symmetric_diff=len(node_names) * 7 + len(prior_graph.get_edges()),
        n_weight_vectors=300,
        n_neighbors=30,
        n_generations=args.generations,
        prob_neighbor_mating=0.9,
        max_replacements=2,
        mutation_prob=0.3,
        mutation_ops_min=2,
        mutation_ops_max=6,
        data=data,
        random_seed=args.seed,
    )

    print(f"运行 MOEA/D ({args.crossover_type}, {args.generations} gens)...")
    MOEAD.run = profiled_run
    moead = MOEAD(config, prior_graph, data, node_names)
    result = moead.run()

    print(f"\n总耗时: {result.runtime:.1f}s")

    # Print per-step timing summary (skip first gen as warmup)
    t = result._timing
    for step in ["cross", "eval", "mutate", "neighbor", "nadir", "other", "total"]:
        vals = t[step][1:]  # skip first gen
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        pct = mean_val / np.mean(t["total"][1:]) * 100
        print(f"  {step:<12}: {mean_val:>6.2f}s ± {std_val:.2f}s ({pct:>5.1f}%)")

    # Save for comparison
    out_path = f"/tmp/profile_{args.crossover_type}.json"
    with open(out_path, "w") as f:
        json.dump({k: [float(x) for x in v] for k, v in t.items()}, f, indent=2)
    print(f"\nTiming saved: {out_path}")


if __name__ == "__main__":
    main()
