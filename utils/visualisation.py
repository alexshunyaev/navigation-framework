import logging
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def _draw_path_and_trajectory(
        ax: plt.Axes,
        path: List[Tuple[int, int]],
        actual_trajectory: np.ndarray,
) -> None:
    """Overlay the A* path, start/goal markers, and local-planner trajectory onto an axes."""
    if path:
        path_arr = np.array(path)
        ax.plot(path_arr[:, 1], path_arr[:, 0], "b--", linewidth=2, label="A* Global Path")
        ax.plot(path_arr[0, 1], path_arr[0, 0], "go", markersize=10, label="Start")
        ax.plot(path_arr[-1, 1], path_arr[-1, 0], "ro", markersize=10, label="Goal")

    if actual_trajectory is not None:
        ax.plot(
            actual_trajectory[:, 1],
            actual_trajectory[:, 0],
            color="crimson", linewidth=2.5, label="Local Planner Trajectory",
        )


def visualize_path(
        grid: np.ndarray,
        path: List[Tuple[int, int]],
        actual_trajectory: np.ndarray = None,
        prob_map=None,
        save_path: str = "experiments/dwa_result.png",
) -> None:
    """
    Visualise navigation results as a two-panel figure.

    Left panel: binary occupancy grid (obstacles clearly visible as black cells).
    Right panel: probabilistic risk field with the same path/trajectory overlay,
                 showing how obstacle uncertainty spreads into neighbouring cells.
    When no prob_map is provided the right panel is omitted.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if prob_map is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(grid, cmap="binary", origin="upper")
        ax.set_title("Navigation Result (A* + Local Planner)")
        _draw_path_and_trajectory(ax, path, actual_trajectory)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
    else:
        fig, (ax_bin, ax_prob) = plt.subplots(1, 2, figsize=(18, 8))

        # --- Left: binary grid ---
        ax_bin.imshow(grid, cmap="binary", origin="upper")
        ax_bin.set_title("Binary Occupancy Grid", fontsize=13)
        _draw_path_and_trajectory(ax_bin, path, actual_trajectory)
        ax_bin.grid(True, alpha=0.3)
        ax_bin.legend(fontsize=11)

        # --- Right: probabilistic risk field ---
        img = ax_prob.imshow(
            prob_map.risk_field,
            cmap="RdYlGn_r",
            vmin=0.0, vmax=1.0,
            origin="upper",
        )
        plt.colorbar(img, ax=ax_prob, fraction=0.046, pad=0.04, label="Obstacle Risk")
        ax_prob.set_title("Probabilistic Risk Field", fontsize=13)
        _draw_path_and_trajectory(ax_prob, path, actual_trajectory)
        ax_prob.grid(True, alpha=0.3)
        ax_prob.legend(fontsize=11)

        fig.suptitle("Navigation Result: A* Global Path + Local Planner", fontsize=15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    logger.info("Plot saved to %s.", save_path)
    plt.close()
