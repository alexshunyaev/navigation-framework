import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


def visualize_path(grid: np.ndarray, path: List[Tuple[int, int]], actual_trajectory: np.ndarray = None):
    """
    Visualize the grid, found A* path, and actual DWA trajectory.
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(grid, cmap='binary')

    if path:
        path = np.array(path)
        plt.plot(path[:, 1], path[:, 0], 'b--', linewidth=2, label='A* Global Path')
        plt.plot(path[0, 1], path[0, 0], 'go', markersize=10, label='Start')
        plt.plot(path[-1, 1], path[-1, 0], 'ro', markersize=10, label='Goal')

    if actual_trajectory is not None:
        plt.plot(actual_trajectory[:, 1], actual_trajectory[:, 0], 'r-', linewidth=3, label='DWA Trajectory')

    plt.grid(True)
    plt.legend(fontsize=12)
    plt.title("Navigation Result (A* + DWA)")
    plt.savefig("experiments/dwa_result.png")
    print("Plot saved to experiments/dwa_result.png")
