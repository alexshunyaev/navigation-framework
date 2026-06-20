import random
import numpy as np

random.seed(42)
np.random.seed(42)

from src.a_star import find_path
from src.dwa import DWAConfig, dwa_control, motion
from utils.visualisation import visualize_path


def generate_random_map(size=20, obstacle_prob=0.3):
    """Generates a random map with obstacles and valid start/goal positions."""
    grid = np.zeros((size, size))
    for r in range(size):
        for c in range(size):
            if random.random() < obstacle_prob:
                grid[r, c] = 1

    def get_random_free_pos():
        while True:
            r = random.randint(0, size - 1)
            c = random.randint(0, size - 1)
            if grid[r, c] == 0:
                return (r, c)

    start_pos = get_random_free_pos()
    goal_pos = get_random_free_pos()
    while goal_pos == start_pos:
        goal_pos = get_random_free_pos()

    return grid, start_pos, goal_pos


def main():
    # Create a random grid with obstacles
    grid, start_pos, goal_pos = generate_random_map(size=20, obstacle_prob=0.2)

    print(f"Start: {start_pos}, Goal: {goal_pos}")

    # 1. Global Planning using A*
    print("Running A* Global Planner...")
    path = find_path(grid, start_pos, goal_pos)
    if not path:
        print("No global path found!")
        return

    print(f"Global path found with {len(path)} steps!")

    # 2. Extract obstacles for Local Planner
    ob = np.argwhere(grid == 1)

    # 3. Setup DWA
    print("Starting DWA Local Planner...")
    config = DWAConfig()

    # Initial state [r, c, theta, v, omega]
    initial_theta = 0.0
    if len(path) > 1:
        initial_theta = np.arctan2(path[1][1] - path[0][1], path[1][0] - path[0][0])

    x = np.array([float(start_pos[0]), float(start_pos[1]), initial_theta, 0.0, 0.0])
    trajectory_history = np.array([x])

    waypoint_idx = 1 if len(path) > 1 else 0

    for i in range(2000):  # max iterations
        goal_waypoint = path[waypoint_idx]

        # Check if we reached the current waypoint (relaxed radius to avoid orbiting)
        dist_to_waypoint = np.hypot(x[0] - goal_waypoint[0], x[1] - goal_waypoint[1])
        if dist_to_waypoint <= 0.8:
            if waypoint_idx < len(path) - 1:
                waypoint_idx += 1
                goal_waypoint = path[waypoint_idx]
            else:
                print("Goal Reached successfully!")
                break

        # Run DWA to get control input
        u, _ = dwa_control(x, config, goal_waypoint, ob)

        # Apply control to simulate motion
        x = motion(x, u, config.dt)
        trajectory_history = np.vstack((trajectory_history, x))

        # Check collision for safety
        if ob.size > 0:
            distances = np.hypot(x[0] - ob[:, 0], x[1] - ob[:, 1])
            if np.min(distances) <= config.robot_radius / 2.0:
                print("Collision Detected! Navigation failed.")
                break
    else:
        print("Max iterations reached. Did not reach goal.")

    print(f"Simulation finished. Actual trajectory length: {len(trajectory_history)} steps.")

    # Visualize the result
    visualize_path(grid, path, trajectory_history)


if __name__ == '__main__':
    main()
