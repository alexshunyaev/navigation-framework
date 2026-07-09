import logging
from math import sqrt
from typing import List, Tuple, Dict

import numpy as np

logger = logging.getLogger(__name__)


def calculate_heuristic(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    """
    Euclidean distance between two grid cells, used as the A* heuristic.

    Euclidean distance never overestimates the true 8-connected cost, so it is
    admissible and keeps A* optimal.
    """
    r1, c1 = pos1
    r2, c2 = pos2
    return sqrt((r2 - r1) ** 2 + (c2 - c1) ** 2)


def get_valid_neighbors(grid: np.ndarray, position: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Get all valid neighbouring cells (8-connected) for a grid position.

    Args:
        grid: 2D numpy array where 0 represents walkable cells and 1 represents obstacles.
        position: Current cell as (row, col).

    Returns:
        List of valid neighbouring cells as (row, col) tuples.
    """
    row, col = position
    rows, cols = grid.shape

    def in_bounds_free(r: int, c: int) -> bool:
        """True if cell (r, c) is inside the grid and not an obstacle."""
        return 0 <= r < rows and 0 <= c < cols and grid[r, c] == 0

    # Orthogonal moves
    orthogonal_moves = [(row + 1, col), (row - 1, col), (row, col + 1), (row, col - 1)]
    neighbors = [(r, c) for r, c in orthogonal_moves if in_bounds_free(r, c)]

    # Diagonal moves: only allowed if both flanking orthogonal cells are free,
    # otherwise the path would cut through the corner of an obstacle.
    diagonal_moves = [
        (row + 1, col + 1), (row - 1, col - 1),
        (row + 1, col - 1), (row - 1, col + 1),
    ]
    for r, c in diagonal_moves:
        if in_bounds_free(r, c) and in_bounds_free(r, col) and in_bounds_free(row, c):
            neighbors.append((r, c))

    return neighbors


def reconstruct_path(goal_node: Dict) -> List[Tuple[int, int]]:
    """
    Reconstruct the path from goal to start by following parent pointers.
    """
    path = []
    current = goal_node

    while current is not None:
        path.append(current['position'])
        current = current['parent']

    return path[::-1]  # Reverse to get path from start to goal
