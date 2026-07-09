import logging
import random
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


def generate_random_map(
        size: int = 20,
        obstacle_prob: float = 0.3,
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    """Generate a random binary occupancy grid with guaranteed free start and goal."""
    grid = np.zeros((size, size))
    for r in range(size):
        for c in range(size):
            if random.random() < obstacle_prob:
                grid[r, c] = 1

    def random_free() -> Tuple[int, int]:
        """Return a uniformly random free (non-obstacle) cell."""
        while True:
            r, c = random.randint(0, size - 1), random.randint(0, size - 1)
            if grid[r, c] == 0:
                return (r, c)

    start = random_free()
    goal = random_free()
    while goal == start:
        goal = random_free()

    logger.info("Map generated: size=%dx%d, obstacle_prob=%.0f%%, start=%s, goal=%s",
                size, size, obstacle_prob * 100, start, goal)
    return grid, start, goal
