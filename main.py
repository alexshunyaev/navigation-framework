import random
import numpy as np

from src.a_star import find_path
from utils.visualisation_a_star import visualize_path

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

# Create a random grid with obstacles
grid, start_pos, goal_pos = generate_random_map(size=20, obstacle_prob=0.3)

# Find the path
path = find_path(grid, start_pos, goal_pos)
if path:
    print(f"Path found with {len(path)} steps!")
    visualize_path(grid, path)
else:
    print("No path found!")\
    