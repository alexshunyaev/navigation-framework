import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from src.prob_map import ProbabilisticMap

logger = logging.getLogger(__name__)


@dataclass
class MPPIConfig:
    """
    Configuration for the Model Predictive Path Integral (MPPI) local planner.

    MPPI replaces DWA's exhaustive grid search over a *single* constant control
    with sampling-based stochastic optimal control over a whole control
    *sequence*. Because each rollout can bend and turn along the horizon (rather
    than tracing a single circular arc), MPPI represents manoeuvres such as
    "turn, then go straight" that DWA structurally cannot — which is exactly what
    is needed to thread cluttered corridors without getting stuck.
    """
    # Kinematic limits (shared convention with DWAConfig)
    max_speed: float = 1.0  # [m/s]
    min_speed: float = -0.2  # [m/s]
    max_yaw_rate: float = 100.0 * math.pi / 180.0  # [rad/s]

    # Integration / horizon
    dt: float = 0.1  # [s]
    horizon: int = 20  # number of control steps (T)

    # Sampling
    num_samples: int = 500  # K rollouts per control step
    temperature: float = 1.0  # lambda — softness of the cost weighting
    sigma_v: float = 0.3  # exploration std on linear velocity [m/s]
    sigma_w: float = 0.5  # exploration std on yaw rate [rad/s]

    # Cost weights. ``goal_cost`` (distance from the horizon endpoint to the
    # local goal) is the primary driver of progress — heading alone only aligns
    # orientation and gives the robot no incentive to actually advance, which
    # otherwise lets the obstacle term freeze it in the lowest-risk cell.
    goal_cost_weight: float = 5.0
    heading_cost_weight: float = 1.0
    obstacle_cost_weight: float = 3.0
    velocity_cost_weight: float = 1.0
    smoothness_cost_weight: float = 0.3
    path_deviation_cost_weight: float = 1.5

    # Penalty added per rolled-out state that violates the collision threshold.
    # Kept large but finite (not inf) so the softmax weighting stays well-defined
    # even when many samples clip an obstacle.
    collision_penalty: float = 1.0e3
    robot_radius: float = 0.3  # [m]


class MPPIPlanner:
    """
    Warm-started MPPI controller.

    Call it exactly like ``dwa_control`` — ``planner(x, config, goal, ob,
    prob_map, global_path)`` — and it returns ``(best_u, best_trajectory)``.
    The nominal control sequence is retained between calls and shifted forward
    by one step (receding horizon), which warm-starts each optimisation and
    yields temporally consistent, smooth commands.
    """

    def __init__(self, config: MPPIConfig, seed: Optional[int] = None):
        """
        Args:
            config: MPPI sampling and cost-weight parameters.
            seed: Optional RNG seed for reproducible sampling.
        """
        self.config = config
        self._rng = np.random.default_rng(seed)
        # Nominal control sequence U = [[v, w], ...], shape (horizon, 2).
        self._nominal = np.zeros((config.horizon, 2))

    # -- rollout ------------------------------------------------------------

    def _rollout(self, x0: np.ndarray, controls: np.ndarray) -> np.ndarray:
        """
        Vectorised differential-drive rollout.

        Args:
            x0: initial state [r, c, theta, v, omega].
            controls: (K, T, 2) sampled control sequences.
        Returns:
            positions: (K, T, 2) predicted (r, c) for every sample and step.
        """
        K = controls.shape[0]
        dt = self.config.dt

        r = np.full(K, x0[0])
        c = np.full(K, x0[1])
        theta = np.full(K, x0[2])

        positions = np.empty((K, self.config.horizon, 2))
        for t in range(self.config.horizon):
            v = controls[:, t, 0]
            w = controls[:, t, 1]
            theta = theta + w * dt
            r = r + v * np.cos(theta) * dt
            c = c + v * np.sin(theta) * dt
            positions[:, t, 0] = r
            positions[:, t, 1] = c
        return positions

    # -- cost ---------------------------------------------------------------

    def _trajectory_costs(
            self,
            positions: np.ndarray,
            controls: np.ndarray,
            goal: Tuple[float, float],
            ob: np.ndarray,
            prob_map: Optional[ProbabilisticMap],
            global_path: Optional[np.ndarray],
    ) -> np.ndarray:
        """Total cost per sampled trajectory, shape (K,)."""
        cfg = self.config
        K, T, _ = positions.shape

        # --- obstacle cost: soft guidance (mean risk) keeps the robot off
        #     obstacles without ever outweighing progress, while the hard
        #     collision penalty (summed) rules out samples that actually breach
        #     the threshold. Summing the soft term over the horizon instead
        #     would let a long safe-but-slightly-risky corridor look worse than
        #     sitting still, freezing the robot. ---
        flat = positions.reshape(-1, 2)
        if prob_map is not None:
            risk = prob_map.get_risk_batch(flat).reshape(K, T)
            soft = (risk / prob_map.collision_threshold) ** 2
            collision = cfg.collision_penalty * (risk >= prob_map.collision_threshold)
        elif len(ob) > 0:
            dx = flat[:, 0][:, None] - ob[:, 0][None, :]
            dy = flat[:, 1][:, None] - ob[:, 1][None, :]
            min_r = np.sqrt(dx * dx + dy * dy).min(axis=1).reshape(K, T)
            soft = 1.0 / np.maximum(min_r, 1e-6)
            collision = cfg.collision_penalty * (min_r <= cfg.robot_radius)
        else:
            soft = np.zeros((K, T))
            collision = np.zeros((K, T))
        obstacle_cost = soft.mean(axis=1) + collision.sum(axis=1)

        # --- goal cost: distance from the horizon endpoint to the local goal ---
        end = positions[:, -1, :]  # (K, 2)
        goal_cost = np.hypot(goal[0] - end[:, 0], goal[1] - end[:, 1])

        # --- heading cost at the terminal state ---
        # heading is the direction of travel over the final step
        prev = positions[:, -2, :] if T > 1 else positions[:, -1, :]
        head = np.arctan2(end[:, 1] - prev[:, 1], end[:, 0] - prev[:, 0])
        desired = np.arctan2(goal[1] - end[:, 1], goal[0] - end[:, 0])
        ang = np.arctan2(np.sin(desired - head), np.cos(desired - head))
        heading_cost = np.abs(ang)

        # --- velocity cost: reward progress toward max speed ---
        velocity_cost = (cfg.max_speed - controls[:, :, 0]).mean(axis=1)

        # --- smoothness: penalise large yaw rates and jerky control changes ---
        smoothness_cost = np.abs(controls[:, :, 1]).mean(axis=1)
        smoothness_cost += np.abs(np.diff(controls[:, :, 1], axis=1)).mean(axis=1)

        # --- path deviation of the terminal state from the global path ---
        if global_path is not None:
            gd = np.hypot(global_path[:, 0][None, :] - end[:, 0][:, None],
                          global_path[:, 1][None, :] - end[:, 1][:, None])
            deviation_cost = gd.min(axis=1)
        else:
            deviation_cost = np.zeros(K)

        return (
                cfg.goal_cost_weight * goal_cost
                + cfg.heading_cost_weight * heading_cost
                + cfg.obstacle_cost_weight * obstacle_cost
                + cfg.velocity_cost_weight * velocity_cost
                + cfg.smoothness_cost_weight * smoothness_cost
                + cfg.path_deviation_cost_weight * deviation_cost
        )

    def __call__(
            self,
            x: np.ndarray,
            config,  # accepted for signature parity with dwa_control; unused
            goal: Tuple[float, float],
            ob: np.ndarray,
            prob_map: Optional[ProbabilisticMap] = None,
            global_path: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run one MPPI optimisation step and return the chosen control and rollout.

        Matches the ``dwa_control`` signature so the two planners are
        interchangeable in :func:`~src.navigation.run_navigation`.

        Args:
            x: Current robot state [r, c, theta, v, omega].
            config: Accepted for signature parity with ``dwa_control``; unused.
            goal: Current lookahead target (r, c).
            ob: Array of obstacle cells, shape (N, 2); used only when no
                ``prob_map`` is supplied.
            prob_map: Optional probabilistic map (Option C) for the smooth
                obstacle cost and hard collision penalty.
            global_path: Optional global path (M, 2); activates the
                path-deviation term (Option D).

        Returns:
            (best_u, best_trajectory): first optimised control [v, omega] and the
            (T+1, 5) rollout of the optimised nominal sequence.
        """
        cfg = self.config
        T = cfg.horizon

        # 1. Sample K control-sequence perturbations around the nominal.
        noise = self._rng.normal(size=(cfg.num_samples, T, 2))
        noise[:, :, 0] *= cfg.sigma_v
        noise[:, :, 1] *= cfg.sigma_w
        controls = self._nominal[None, :, :] + noise
        np.clip(controls[:, :, 0], cfg.min_speed, cfg.max_speed, out=controls[:, :, 0])
        np.clip(controls[:, :, 1], -cfg.max_yaw_rate, cfg.max_yaw_rate, out=controls[:, :, 1])

        # 2. Roll out and score every sample.
        positions = self._rollout(x, controls)
        costs = self._trajectory_costs(positions, controls, goal, ob, prob_map, global_path)

        # 3. Path-integral weights: softmin over trajectory costs.
        beta = costs.min()
        weights = np.exp(-(costs - beta) / cfg.temperature)
        weights /= weights.sum() + 1e-9

        # 4. Weighted update of the nominal sequence, then clip to limits.
        self._nominal = self._nominal + np.einsum("k,ktd->td", weights, noise)
        np.clip(self._nominal[:, 0], cfg.min_speed, cfg.max_speed, out=self._nominal[:, 0])
        np.clip(self._nominal[:, 1], -cfg.max_yaw_rate, cfg.max_yaw_rate, out=self._nominal[:, 1])

        best_u = self._nominal[0].copy()

        # 5. Roll out the optimised nominal for visualisation / inspection.
        best_positions = self._rollout(x, self._nominal[None, :, :])[0]
        best_trajectory = np.zeros((T + 1, 5))
        best_trajectory[0] = x
        best_trajectory[1:, 0:2] = best_positions
        best_trajectory[1:, 3:5] = self._nominal

        # 6. Receding horizon: shift the nominal forward, repeat the last action.
        self._nominal = np.vstack([self._nominal[1:], self._nominal[-1:]])

        return best_u, best_trajectory
