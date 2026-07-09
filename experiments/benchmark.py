"""
Reproducible experimental evaluation: MPPI vs. DWA local planners.

Generates a batch of seeded random maps, plans a global A* path on the inflated
risk grid for each, then runs both local planners along that path and reports
the goal-reaching success rate and average step count. Run from the repo root:

    python -m experiments.benchmark
    python -m experiments.benchmark --maps 25 --min-path 8

The random seed makes the whole batch deterministic so the numbers quoted in the
report can be reproduced exactly.
"""
import argparse
import logging
import random

import numpy as np

from src.a_star import find_path
from src.dwa import DWAConfig
from src.mppi import MPPIConfig, MPPIPlanner
from src.navigation import run_navigation
from src.prob_map import ProbabilisticMap
from utils.map_generator import generate_random_map


def _solvable_map(seed: int, size: int, obstacle_prob: float, min_path: int):
    """Seed the RNG and generate a map with a non-trivial, solvable global path."""
    random.seed(seed)
    np.random.seed(seed)
    for _ in range(50):
        grid, start, goal = generate_random_map(size=size, obstacle_prob=obstacle_prob)
        prob_map = ProbabilisticMap(grid, sigma=0.8, collision_threshold=0.5)
        planning_grid = prob_map.planning_grid()
        if planning_grid[start] == 1 or planning_grid[goal] == 1:
            continue
        path = find_path(planning_grid, start, goal)
        if path and len(path) >= min_path:
            return grid, start, goal, prob_map, path
    return None


def run_benchmark(num_maps: int, size: int, obstacle_prob: float,
                  min_path: int, max_iterations: int) -> None:
    """
    Run MPPI and DWA over ``num_maps`` seeded maps and print a summary table.

    For every solvable map both planners follow the same A* path and share the
    same probabilistic cost, so the reported goal-reaching rate and average step
    count isolate the effect of the local optimiser alone.
    """
    results = {"MPPI": [], "DWA": []}
    evaluated = 0
    seed = 0
    while evaluated < num_maps:
        m = _solvable_map(seed, size, obstacle_prob, min_path)
        seed += 1
        if m is None:
            continue
        grid, start, goal, prob_map, path = m
        evaluated += 1

        planner = MPPIPlanner(MPPIConfig(), seed=0)
        traj, ok = run_navigation(grid, start, goal, DWAConfig(), prob_map, path,
                                  planner=planner, max_iterations=max_iterations)
        results["MPPI"].append((ok, len(traj)))

        dwa_config = DWAConfig(smoothness_cost_weight=1.0, path_deviation_cost_weight=0.5)
        traj, ok = run_navigation(grid, start, goal, dwa_config, prob_map, path,
                                  planner=None, max_iterations=max_iterations)
        results["DWA"].append((ok, len(traj)))

    print(f"\nEvaluated {evaluated} maps "
          f"({size}x{size}, obstacle_prob={obstacle_prob}, min_path={min_path})\n")
    print(f"{'Planner':<8}{'Reached goal':<16}{'Avg steps (reached)':<20}")
    print("-" * 44)
    for name, runs in results.items():
        reached = [n for ok, n in runs if ok]
        avg = f"{np.mean(reached):.1f}" if reached else "n/a"
        print(f"{name:<8}{f'{len(reached)}/{len(runs)}':<16}{avg:<20}")


def main() -> None:
    """Parse command-line options and launch the benchmark."""
    parser = argparse.ArgumentParser(description="Benchmark MPPI vs DWA local planners.")
    parser.add_argument("--maps", type=int, default=25, help="number of maps to evaluate")
    parser.add_argument("--size", type=int, default=20, help="grid side length")
    parser.add_argument("--obstacle-prob", type=float, default=0.3, help="obstacle density")
    parser.add_argument("--min-path", type=int, default=8, help="minimum A* path length")
    parser.add_argument("--max-iterations", type=int, default=600, help="step budget per run")
    args = parser.parse_args()

    logging.disable(logging.CRITICAL)  # keep the table clean
    run_benchmark(args.maps, args.size, args.obstacle_prob,
                  args.min_path, args.max_iterations)


if __name__ == "__main__":
    main()
