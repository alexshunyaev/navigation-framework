import numpy as np

from src.a_star import find_path
from utils.visualisation_a_star import visualize_path

# Create a sample grid
grid = np.zeros((20, 20))  # 20x20 grid, all free space initially
# Add some obstacles
grid[5:15, 10] = 1  # Vertical wall
grid[5, 5:15] = 1   # Horizontal wall
# Define start and goal positions
start_pos = (2, 2)
goal_pos = (18, 18)
# Find the path
path = find_path(grid, start_pos, goal_pos)
if path:
    print(f"Path found with {len(path)} steps!")
    visualize_path(grid, path)
else:
    print("No path found!")\
    