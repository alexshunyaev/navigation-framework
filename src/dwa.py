import math
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class DWAConfig:
    """
    Configuration parameters for the Dynamic Window Approach (DWA).
    We use a dataclass to keep configuration modular, easily tunable, and cleanly separated from the logic.
    """
    max_speed: float = 1.0  # [m/s]
    min_speed: float = -0.2  # [m/s]
    max_yaw_rate: float = 100.0 * math.pi / 180.0  # [rad/s]
    max_accel: float = 1.0  # [m/ss]
    max_delta_yaw_rate: float = 100.0 * math.pi / 180.0  # [rad/ss]
    v_resolution: float = 0.02  # [m/s]
    yaw_rate_resolution: float = 0.5 * math.pi / 180.0  # [rad/s]
    dt: float = 0.1  # [s] Time tick for motion prediction
    predict_time: float = 1.5  # [s] Forward simulation horizon
    
    # Cost function weights: G(v, w) = alpha*heading + beta*obstacle + gamma*velocity
    heading_cost_weight: float = 2.0   # alpha
    obstacle_cost_weight: float = 0.5   # beta
    velocity_cost_weight: float = 2.0   # gamma
    
    robot_radius: float = 0.3  # [m] Safety boundary


def motion(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    """
    Simulate the robot's motion using simple differential drive kinematics.
    State x: [x, y, theta, v, omega]
    Control u: [v, omega]
    """
    x[2] += u[1] * dt
    x[0] += u[0] * math.cos(x[2]) * dt
    x[1] += u[0] * math.sin(x[2]) * dt
    x[3] = u[0]
    x[4] = u[1]
    return x


def calc_dynamic_window(x: np.ndarray, config: DWAConfig) -> List[float]:
    """
    Calculate the dynamic window (reachable velocities) based on physical limitations.
    This restricts the search space to velocities the robot can actually achieve in the next time step.
    Returns: [v_min, v_max, omega_min, omega_max]
    """
    # System limitations
    Vs = [config.min_speed, config.max_speed,
          -config.max_yaw_rate, config.max_yaw_rate]

    # Reachable velocities based on current acceleration
    Vd = [x[3] - config.max_accel * config.dt,
          x[3] + config.max_accel * config.dt,
          x[4] - config.max_delta_yaw_rate * config.dt,
          x[4] + config.max_delta_yaw_rate * config.dt]

    # Intersection of system limitations and reachability bounds
    dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
          max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]
    return dw


def predict_trajectory(x_init: np.ndarray, v: float, w: float, config: DWAConfig) -> np.ndarray:
    """
    Predict a trajectory by rolling forward the kinematic model assuming constant velocity (v, w).
    """
    x = np.array(x_init)
    trajectory = np.array(x)
    time = 0.0
    while time <= config.predict_time:
        x = motion(x, np.array([v, w]), config.dt)
        trajectory = np.vstack((trajectory, x))
        time += config.dt
    return trajectory


def calc_obstacle_cost(trajectory: np.ndarray, ob: np.ndarray, config: DWAConfig) -> float:
    """
    Calculate the obstacle cost by finding the minimum distance to any obstacle.
    We return 1.0 / min_distance to penalize trajectories that get too close to obstacles.
    """
    if len(ob) == 0:
        return 0.0
    
    ox = ob[:, 0]
    oy = ob[:, 1]
    
    # Broadcast subtraction to find distances from all trajectory points to all obstacles
    dx = trajectory[:, 0] - ox[:, None]
    dy = trajectory[:, 1] - oy[:, None]
    r = np.hypot(dx, dy)
    
    min_r = np.min(r)
    if min_r <= config.robot_radius:
        return float("inf")  # Collision!
        return float("inf")  # Collision!
    
    return 1.0 / min_r


def calc_heading_cost(trajectory: np.ndarray, goal: Tuple[float, float]) -> float:
    """
    Calculate the heading cost. It measures the alignment with the goal at the end of the trajectory.
    Smaller angles mean we are pointed directly at the target.
    """
    dx = goal[0] - trajectory[-1, 0]
    dy = goal[1] - trajectory[-1, 1]
    error_angle = math.atan2(dy, dx)
    cost_angle = error_angle - trajectory[-1, 2]
    # Normalize the angle difference
    cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))
    return cost


def dwa_control(x: np.ndarray, config: DWAConfig, goal: Tuple[float, float], ob: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dynamic Window Approach control loop. 
    It evaluates candidate velocities (v, w) and returns the optimal control and its predicted trajectory.
    """
    dw = calc_dynamic_window(x, config)

    min_cost = float("inf")
    best_u = np.array([0.0, 0.0])
    best_trajectory = np.array([x])

    # Grid search over the dynamic window to evaluate candidate trajectories
    for v in np.arange(dw[0], dw[1], config.v_resolution):
        for w in np.arange(dw[2], dw[3], config.yaw_rate_resolution):
            trajectory = predict_trajectory(x, v, w, config)

            # Evaluate objective function (Cost G)
            heading_cost = config.heading_cost_weight * calc_heading_cost(trajectory, goal)
            ob_cost = config.obstacle_cost_weight * calc_obstacle_cost(trajectory, ob, config)
            # Velocity cost: penalty for slow speeds, reward for high speeds
            vel_cost = config.velocity_cost_weight * (config.max_speed - trajectory[-1, 3])

            final_cost = heading_cost + ob_cost + vel_cost
            
            if final_cost < min_cost:
                min_cost = final_cost
                best_u = np.array([v, w])
                best_trajectory = trajectory
                
    return best_u, best_trajectory
