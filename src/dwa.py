import logging
import math
from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np

from src.prob_map import ProbabilisticMap

logger = logging.getLogger(__name__)


@dataclass
class DWAConfig:
    """
    Configuration parameters for the Dynamic Window Approach (DWA).

    Core weights (alpha, beta, gamma) replicate the original objective function.
    Extension weights (smoothness, path_deviation) activate the multi-objective
    terms from Option D. Set use_prob_map=True to switch the obstacle cost to
    the probabilistic field from Option C.
    """
    # Kinematic limits
    max_speed: float = 1.0  # [m/s]
    min_speed: float = -0.2  # [m/s]
    max_yaw_rate: float = 100.0 * math.pi / 180.0  # [rad/s]
    max_accel: float = 1.0  # [m/s²]
    max_delta_yaw_rate: float = 100.0 * math.pi / 180.0  # [rad/s²]

    # Search resolution
    v_resolution: float = 0.02  # [m/s]
    yaw_rate_resolution: float = 0.5 * math.pi / 180.0  # [rad/s]

    # Simulation horizon
    dt: float = 0.1  # [s]
    predict_time: float = 1.5  # [s]

    # Core objective weights: G = alpha*heading + beta*obstacle + gamma*velocity
    heading_cost_weight: float = 2.0  # alpha
    obstacle_cost_weight: float = 3.0  # beta
    velocity_cost_weight: float = 2.0  # gamma

    # Distance from the trajectory endpoint to the goal. The classic heading
    # term only rewards orientation, giving no incentive to actually close
    # distance; without this term the normalised obstacle cost can freeze the
    # robot in the safest nearby cell instead of advancing.
    goal_cost_weight: float = 3.0

    # Option D — multi-objective extension weights
    smoothness_cost_weight: float = 1.0  # penalise high angular velocity
    path_deviation_cost_weight: float = 0.5  # penalise drift from global path

    robot_radius: float = 0.3  # [m]


# ---------------------------------------------------------------------------
# Kinematic model
# ---------------------------------------------------------------------------

def motion(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    """
    Differential-drive kinematic update.
    State x: [r, c, theta, v, omega]
    Control u: [v, omega]
    """
    x = x.copy()
    x[2] += u[1] * dt
    x[0] += u[0] * math.cos(x[2]) * dt
    x[1] += u[0] * math.sin(x[2]) * dt
    x[3] = u[0]
    x[4] = u[1]
    return x


def calc_dynamic_window(x: np.ndarray, config: DWAConfig) -> List[float]:
    """
    Intersect physical speed limits with the set of velocities reachable
    within one time step to produce the dynamic window.
    Returns [v_min, v_max, omega_min, omega_max].
    """
    Vs = [config.min_speed, config.max_speed,
          -config.max_yaw_rate, config.max_yaw_rate]
    Vd = [x[3] - config.max_accel * config.dt,
          x[3] + config.max_accel * config.dt,
          x[4] - config.max_delta_yaw_rate * config.dt,
          x[4] + config.max_delta_yaw_rate * config.dt]
    return [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
            max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]


def predict_trajectory(x_init: np.ndarray, v: float, w: float,
                       config: DWAConfig) -> np.ndarray:
    """Roll out the kinematic model for predict_time seconds at constant (v, w)."""
    u = np.array([v, w])
    x = x_init.copy()
    states = [x]
    t = 0.0
    while t <= config.predict_time:
        x = motion(x, u, config.dt)
        states.append(x)
        t += config.dt
    return np.array(states)


# ---------------------------------------------------------------------------
# Cost functions
# ---------------------------------------------------------------------------

def calc_obstacle_cost(trajectory: np.ndarray, ob: np.ndarray,
                       config: DWAConfig) -> float:
    """
    Binary obstacle cost: inf if the trajectory enters the safety radius,
    otherwise 1 / min_distance (larger penalty for closer passes).
    """
    if len(ob) == 0:
        return 0.0
    dx = trajectory[:, 0] - ob[:, 0, np.newaxis]
    dy = trajectory[:, 1] - ob[:, 1, np.newaxis]
    r = np.hypot(dx, dy)
    min_r = np.min(r)
    if min_r <= config.robot_radius:
        return float("inf")
    return 1.0 / min_r


def calc_heading_cost(trajectory: np.ndarray, goal: Tuple[float, float]) -> float:
    """
    Angular deviation between the robot's heading at the end of the trajectory
    and the direction toward the goal. Zero means perfectly aligned.
    """
    dx = goal[0] - trajectory[-1, 0]
    dy = goal[1] - trajectory[-1, 1]
    error_angle = math.atan2(dy, dx)
    cost_angle = error_angle - trajectory[-1, 2]
    return abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))


def calc_smoothness_cost(trajectory: np.ndarray) -> float:
    """
    Option D — smoothness objective.
    Penalises large angular velocities to discourage erratic, jerky turning.
    """
    return float(np.mean(np.abs(trajectory[:, 4])))


def calc_path_deviation_cost(trajectory: np.ndarray,
                             global_path: np.ndarray) -> float:
    """
    Option D — path-consistency objective.
    Measures how far the trajectory endpoint drifts from the nearest point
    on the global reference path, encouraging the robot to stay on course.
    """
    end = trajectory[-1, :2]
    dists = np.hypot(global_path[:, 0] - end[0], global_path[:, 1] - end[1])
    return float(np.min(dists))


# ---------------------------------------------------------------------------
# Main DWA control loop
# ---------------------------------------------------------------------------

def dwa_control(
        x: np.ndarray,
        config: DWAConfig,
        goal: Tuple[float, float],
        ob: np.ndarray,
        prob_map: Optional[ProbabilisticMap] = None,
        global_path: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dynamic Window Approach control step.

    Evaluates every (v, omega) pair in the dynamic window, scores each
    predicted trajectory with the composite objective function, and returns
    the lowest-cost control and its associated trajectory.

    Args:
        x: Current robot state [r, c, theta, v, omega].
        config: DWA parameters and cost weights.
        goal: Current target waypoint (r, c).
        ob: Array of obstacle positions, shape (N, 2).
        prob_map: Optional probabilistic map (Option C). When provided,
                  replaces the discrete obstacle cost with a smooth risk field.
        global_path: Optional global path as (M, 2) array (Option D). When
                     provided, activates the path-deviation penalty term.

    Returns:
        (best_u, best_trajectory): Optimal control [v, omega] and its rollout.
    """
    dw = calc_dynamic_window(x, config)

    # -- Pass 1: roll out every admissible (v, omega) and collect raw cost terms.
    #    The terms live on incompatible scales (heading in radians, clearance in
    #    [0, 1], velocity in m/s, deviation in cells), so multiplying raw weights
    #    against them lets whichever term is numerically largest dominate. We
    #    therefore normalise each term across the candidate set before weighting,
    #    as in the original Fox et al. formulation.
    candidates: List[dict] = []
    for v in np.arange(dw[0], dw[1], config.v_resolution):
        for w in np.arange(dw[2], dw[3], config.yaw_rate_resolution):
            trajectory = predict_trajectory(x, v, w, config)

            if prob_map is not None:
                ob_cost = prob_map.calc_trajectory_cost(trajectory)
            else:
                ob_cost = calc_obstacle_cost(trajectory, ob, config)

            if math.isinf(ob_cost):
                continue  # trajectory collides — discard outright

            candidates.append({
                "u": np.array([v, w]),
                "trajectory": trajectory,
                "heading": calc_heading_cost(trajectory, goal),
                "goal": math.hypot(goal[0] - trajectory[-1, 0], goal[1] - trajectory[-1, 1]),
                "obstacle": ob_cost,
                "velocity": config.max_speed - trajectory[-1, 3],
                "smoothness": calc_smoothness_cost(trajectory),
                "deviation": (calc_path_deviation_cost(trajectory, global_path)
                              if global_path is not None else 0.0),
            })

    if not candidates:
        # Entire dynamic window is blocked — signal a stop so the caller can
        # trigger a recovery behaviour instead of driving into an obstacle.
        return np.array([0.0, 0.0]), x[np.newaxis, :]

    def _normaliser(key: str):
        """Return a function that normalises one cost term by its sum over all candidates."""
        total = sum(c[key] for c in candidates)
        return (lambda val: val / total) if total > 1e-9 else (lambda val: 0.0)

    norm = {k: _normaliser(k)
            for k in ("heading", "goal", "obstacle", "velocity", "smoothness", "deviation")}

    # -- Pass 2: score the normalised, weighted objective and pick the minimum.
    min_cost = float("inf")
    best = candidates[0]
    for c in candidates:
        final_cost = (
                config.heading_cost_weight * norm["heading"](c["heading"])
                + config.goal_cost_weight * norm["goal"](c["goal"])
                + config.obstacle_cost_weight * norm["obstacle"](c["obstacle"])
                + config.velocity_cost_weight * norm["velocity"](c["velocity"])
                + config.smoothness_cost_weight * norm["smoothness"](c["smoothness"])
                + config.path_deviation_cost_weight * norm["deviation"](c["deviation"])
        )
        if final_cost < min_cost:
            min_cost = final_cost
            best = c

    return best["u"], best["trajectory"]
