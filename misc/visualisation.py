import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Get full path safely regardless of where you run the script
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
challenge_path = os.path.join(base_dir, "arc-agi_evaluation_challenges.json")
solution_path = os.path.join(base_dir, "arc-agi_evaluation_solutions.json")

# Load JSON files
with open(challenge_path, "r") as f:
    challenges = json.load(f)

with open(solution_path, "r") as f:
    solutions = json.load(f)

# Helper to plot a grid
def show_grid(grid, title=""):
    plt.imshow(np.array(grid), cmap="tab20", vmin=0, vmax=9)
    plt.title(title)
    plt.axis("off")

# Visualize the first task
task_id = list(challenges.keys())[0]
task = challenges[task_id]
solution = solutions[task_id]

print(f"Showing task: {task_id}")
train_pairs = task["train"]
test_inputs = task["test"]
test_outputs = solution  # this is a list of test output grids

# Plot the train examples
for i, pair in enumerate(train_pairs):
    plt.figure(figsize=(4, 2))
    plt.subplot(1, 2, 1)
    show_grid(pair["input"], title=f"Train Input {i}")
    plt.subplot(1, 2, 2)
    show_grid(pair["output"], title=f"Train Output {i}")
    plt.tight_layout()
    plt.show()

# Plot the test inputs + actual outputs
for i, test in enumerate(test_inputs):
    plt.figure(figsize=(4, 2))
    plt.subplot(1, 2, 1)
    show_grid(test["input"], title=f"Test Input {i}")
    plt.subplot(1, 2, 2)
    show_grid(solution[i], title=f"Ground Truth Output {i}")
    plt.tight_layout()
    plt.show()
