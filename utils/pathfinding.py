import numpy as np
from typing import List, Tuple, Dict
from math import sqrt


def calculate_heuristic(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    """
    Calculate the estimated distance between two points using Euclidean distance.
    """
    x1, y1 = pos1
    x2, y2 = pos2
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def get_valid_neighbors(grid: np.ndarray, position: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Get all valid neighboring positions in the grid.

    Args:
        grid: 2D numpy array where 0 represents walkable cells and 1 represents obstacles
        position: Current position (x, y)

    Returns:
        List of valid neighboring positions
    """
    x, y = position
    rows, cols = grid.shape

    # All possible moves (including diagonals)
    possible_moves = [
        (x + 1, y), (x - 1, y),  # Right, Left
        (x, y + 1), (x, y - 1),  # Up, Down
        (x + 1, y + 1), (x - 1, y - 1),  # Diagonal moves
        (x + 1, y - 1), (x - 1, y + 1)
    ]

    return [
        (nx, ny) for nx, ny in possible_moves
        if 0 <= nx < rows and 0 <= ny < cols  # Within grid bounds
           and grid[nx, ny] == 0  # Not an obstacle
    ]


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
