import math

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def plot_bins(bins, box_dims, bin_width, bin_height, fitness_value, save_to_file=True):
    if save_to_file:
        colors = plt.colormaps.get_cmap("tab20").resampled(len(box_dims))
        num_bins = len(bins)
        grid_size = math.ceil(math.sqrt(num_bins))
        fig, axs = plt.subplots(grid_size, grid_size, figsize=(15, 15))
        axs = np.atleast_2d(axs)  # Ensure axs is always 2D

        for i, ax in enumerate(axs.flatten()):
            if i < num_bins:
                bin_num = list(bins.keys())[i]
                ax.set_xlim(0, bin_width)
                ax.set_ylim(0, bin_height)
                ax.set_title(f"Bin {bin_num}")
                ax.set_aspect("equal")
                ax.invert_yaxis()
                for box in bins[bin_num]:
                    box_id, x, y, rotation = box
                    if rotation == 0:
                        box_width, box_height = box_dims[box_id]
                    else:
                        box_height, box_width = box_dims[box_id]
                    rect = patches.Rectangle(
                        (x, y),
                        box_width,
                        box_height,
                        linewidth=1,
                        edgecolor="r",
                        facecolor=colors(box_id),
                    )
                    ax.add_patch(rect)
                    ax.text(
                        x + box_width / 2,
                        y + box_height / 2,
                        str(box_id),
                        ha="center",
                        va="center",
                    )
            else:
                ax.axis("off")

        plt.suptitle(f"Best Fitness Value: {fitness_value:.2f}", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)  # Adjust to make room for the suptitle
        plt.savefig("solution.png")
