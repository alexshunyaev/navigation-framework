import logging
import os

from src.a_star import find_path
from src.dwa import DWAConfig
from src.mppi import MPPIConfig, MPPIPlanner
from src.navigation import run_navigation
from src.prob_map import ProbabilisticMap
from utils.map_generator import generate_random_map
from utils.visualisation import visualize_path

logger = logging.getLogger(__name__)

# MPPI is the operational local planner. "dwa" is retained only as a comparison
# baseline (see experiments/benchmark.py and the report's failure-case analysis);
# it is not used to drive the robot in normal operation.
PLANNER = "mppi"


def main() -> None:
    """
    Run the full navigation pipeline once and save a result figure.

    Generates a random map, builds the probabilistic risk field, plans a global
    A* path on the inflated planning grid, drives the robot along it with the
    configured local planner (MPPI by default), and visualises the outcome.
    """
    os.makedirs("experiments", exist_ok=True)

    # Retry map generation until start/goal are free on the *inflated* planning
    # grid and A* finds a path there — this guarantees the global path only uses
    # corridors the local planner also considers safe.
    for attempt in range(50):
        grid, start_pos, goal_pos = generate_random_map(size=20, obstacle_prob=0.3)
        prob_map = ProbabilisticMap(grid, sigma=0.8, collision_threshold=0.5)
        planning_grid = prob_map.planning_grid()

        if planning_grid[start_pos] == 1 or planning_grid[goal_pos] == 1:
            continue  # start/goal fall inside the inflated obstacle margin
        path = find_path(planning_grid, start_pos, goal_pos)
        if path:
            break
    else:
        logger.error("Could not generate a solvable map after 50 attempts. Aborting.")
        return

    logger.info("Probabilistic map built (sigma=0.8, threshold=0.5); A* on inflated grid.")

    if PLANNER == "mppi":
        # MPPI carries its own MPPIConfig; the DWAConfig here only supplies the
        # integration step (config.dt) used by the navigation loop's motion model.
        config = DWAConfig()
        planner = MPPIPlanner(MPPIConfig(), seed=0)
        logger.info("Using MPPI local planner (%d samples, horizon %d).",
                    planner.config.num_samples, planner.config.horizon)
    else:
        config = DWAConfig(smoothness_cost_weight=1.0, path_deviation_cost_weight=0.5)
        planner = None  # defaults to dwa_control
        logger.info("Using DWA local planner.")

    trajectory, _ = run_navigation(
        grid, start_pos, goal_pos, config, prob_map, path, planner=planner,
    )

    visualize_path(grid, path, trajectory, prob_map=prob_map,
                   save_path="experiments/dwa_result.png")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
