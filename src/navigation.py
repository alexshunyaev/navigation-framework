import logging
from typing import Callable, List, Optional, Tuple

import numpy as np

from src.dwa import DWAConfig, dwa_control, motion
from src.prob_map import ProbabilisticMap

logger = logging.getLogger(__name__)

# A planner has the same call signature as ``dwa_control``. ``MPPIPlanner`` is
# callable with this signature too, so either can be dropped in here.
Planner = Callable[..., Tuple[np.ndarray, np.ndarray]]


def _lookahead_goal(
        x: np.ndarray,
        path_arr: np.ndarray,
        lookahead: int,
) -> Tuple[Tuple[float, float], int]:
    """
    Pure-pursuit "carrot" goal selection.

    Feeding the planner the immediate next waypoint (often a single cell away)
    makes the heading term jitter and the robot overshoot. Instead we find the
    path point closest to the robot and aim a fixed number of waypoints further
    along, giving the planner a stable target down the corridor.

    Returns the carrot (r, c) and the index of the closest path point.
    """
    dists = np.hypot(path_arr[:, 0] - x[0], path_arr[:, 1] - x[1])
    nearest = int(np.argmin(dists))
    carrot_idx = min(nearest + lookahead, len(path_arr) - 1)
    return (path_arr[carrot_idx, 0], path_arr[carrot_idx, 1]), nearest


def run_navigation(
        grid: np.ndarray,
        start_pos: Tuple[int, int],
        goal_pos: Tuple[int, int],
        config: DWAConfig,
        prob_map: ProbabilisticMap,
        path: List[Tuple[int, int]],
        planner: Optional[Planner] = None,
        max_iterations: int = 2000,
        goal_capture_radius: float = 0.4,
        lookahead: int = 3,
) -> Tuple[np.ndarray, bool]:
    """
    Execute a local planner along an A* global path.

    Uses the probabilistic map (Option C) for smooth obstacle cost and passes
    the full global path to activate the path-deviation penalty (Option D).
    A pure-pursuit lookahead point is fed to the planner as the local goal.

    Args:
        planner: local-planning step. Defaults to :func:`dwa_control`; pass an
            :class:`~src.mppi.MPPIPlanner` instance to use MPPI instead.

    Returns:
        trajectory: (N, 5) array of robot states [r, c, theta, v, omega].
        reached_goal: True if the robot captured the final path point.
    """
    if planner is None:
        planner = dwa_control

    ob = np.argwhere(grid == 1)
    global_path_arr = np.array(path, dtype=float)
    goal_point = global_path_arr[-1]

    initial_theta = 0.0
    if len(path) > 1:
        initial_theta = np.arctan2(
            path[1][1] - path[0][1],
            path[1][0] - path[0][0],
        )

    x = np.array([float(start_pos[0]), float(start_pos[1]), initial_theta, 0.0, 0.0])
    trajectory = x[np.newaxis, :]
    reached_goal = False

    for step in range(max_iterations):
        if np.hypot(x[0] - goal_point[0], x[1] - goal_point[1]) <= goal_capture_radius:
            logger.info("Goal reached at step %d.", step)
            reached_goal = True
            break

        local_goal, nearest = _lookahead_goal(x, global_path_arr, lookahead)

        u, _ = planner(
            x, config, local_goal, ob,
            prob_map=prob_map,
            global_path=global_path_arr,
        )
        x = motion(x, u, config.dt)
        trajectory = np.vstack((trajectory, x))

        if prob_map.get_risk(x[0], x[1]) >= prob_map.collision_threshold:
            logger.warning("Collision detected via risk field at step %d. Aborting.", step)
            break
    else:
        logger.warning("Maximum iterations (%d) reached without finding goal.", max_iterations)

    logger.info("Navigation finished: %d steps, reached_goal=%s.", len(trajectory), reached_goal)
    return trajectory, reached_goal
