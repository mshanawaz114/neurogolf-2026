"""
visualize.py — Display ARC-AGI task grids.

Usage:
    python utils/visualize.py --task tasks/task001.json
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ARC-AGI standard 10-colour palette
ARC_COLORS = [
    "#000000",  # 0 black
    "#0074D9",  # 1 blue
    "#FF4136",  # 2 red
    "#2ECC40",  # 3 green
    "#FFDC00",  # 4 yellow
    "#AAAAAA",  # 5 grey
    "#F012BE",  # 6 magenta
    "#FF851B",  # 7 orange
    "#7FDBFF",  # 8 light blue
    "#870C25",  # 9 dark red
]
CMAP = mcolors.ListedColormap(ARC_COLORS)


def plot_grid(ax, grid, title=""):
    ax.imshow(grid, cmap=CMAP, vmin=0, vmax=9, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def visualize_task(task_path: str):
    with open(task_path) as f:
        task = json.load(f)

    train_pairs = task["train"]
    test_pairs = task.get("test", [])
    all_pairs = [("Train", train_pairs), ("Test", test_pairs)]

    for split_name, pairs in all_pairs:
        if not pairs:
            continue
        n = len(pairs)
        fig, axes = plt.subplots(n, 2, figsize=(4, 2 * n))
        if n == 1:
            axes = [axes]
        fig.suptitle(f"{split_name} pairs — {task_path}", fontsize=10)
        for i, pair in enumerate(pairs):
            plot_grid(axes[i][0], np.array(pair["input"]), f"{split_name} {i+1} Input")
            if "output" in pair:
                plot_grid(axes[i][1], np.array(pair["output"]), f"{split_name} {i+1} Output")
            else:
                axes[i][1].axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, help="Path to task JSON file")
    args = parser.parse_args()
    visualize_task(args.task)
