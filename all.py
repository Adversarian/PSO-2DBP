import math

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


class Particle:
    def __init__(
        self, num_boxes, bin_width, bin_height, box_dims, initial_position=None
    ):
        self.num_boxes = num_boxes
        self.bin_width = bin_width
        self.bin_height = bin_height
        self.box_dims = box_dims  # List of tuples (box_width, box_height)
        self.position = self.initialize_position(initial_position)
        self.velocity = np.zeros_like(self.position, dtype=float)
        self.best_position = self.position.copy()
        self.best_fitness = float("inf")

    def initialize_position(self, initial_position):
        if initial_position is not None:
            return np.array(initial_position, dtype=object)
        # Random initialization if no initial position provided
        position = []
        for i in range(self.num_boxes):
            bin_num = np.random.randint(0, 10)
            rotation = np.random.choice([0, 1])
            if rotation == 0:
                w, h = self.box_dims[i]
            else:
                h, w = self.box_dims[i]
            x = np.random.uniform(0, self.bin_width - w)
            y = np.random.uniform(0, self.bin_height - h)
            position.append((bin_num, x, y, rotation))
        return np.array(position, dtype=object)


def fitness_function(position, bin_width, bin_height, box_dims):
    bins = {}
    for i, (bin_num, x, y, rotation) in enumerate(position):
        if bin_num not in bins:
            bins[bin_num] = []
        if rotation == 0:
            w, h = box_dims[i]
        else:
            h, w = box_dims[i]
        bins[bin_num].append((x, y, w, h))

    num_bins_used = len(bins)
    penalty = 0

    for bin_num, boxes in bins.items():
        for i, (x1, y1, w1, h1) in enumerate(boxes):
            if x1 + w1 > bin_width or y1 + h1 > bin_height:
                penalty += 1
            for j, (x2, y2, w2, h2) in enumerate(boxes):
                if i != j:
                    if not (
                        x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1
                    ):
                        penalty += 1

    return num_bins_used + penalty * 1000


def update_velocity(particle, global_best_position, w, c1, c2):
    r1, r2 = np.random.rand(2)
    cognitive_velocity = (
        c1 * r1 * (particle.best_position[:, 1:] - particle.position[:, 1:])
    )
    social_velocity = c2 * r2 * (global_best_position[:, 1:] - particle.position[:, 1:])
    particle.velocity[:, 1:] = (
        w * particle.velocity[:, 1:] + cognitive_velocity + social_velocity
    )


def update_position(particle):
    particle.position[:, 1:] += particle.velocity[:, 1:]
    for i, (bin_num, x, y, rotation) in enumerate(particle.position):
        if rotation == 0:
            box_width, box_height = particle.box_dims[i]
        else:
            box_height, box_width = particle.box_dims[i]
        particle.position[i][1] = max(0, min(particle.bin_width - box_width, x))
        particle.position[i][2] = max(0, min(particle.bin_height - box_height, y))


def best_fit_decreasing(num_boxes, box_dims, bin_width, bin_height):
    boxes = sorted(enumerate(box_dims), key=lambda x: max(x[1]), reverse=True)
    bins = []
    position = []

    for idx, (box_width, box_height) in boxes:
        best_bin = None
        min_space_left = float("inf")
        for bin_num, bin in enumerate(bins):
            if can_place_in_bin(bin, box_width, box_height, bin_width, bin_height):
                space_left = calculate_space_left(
                    bin, box_width, box_height, bin_width, bin_height
                )
                if space_left < min_space_left:
                    min_space_left = space_left
                    best_bin = bin_num
            if can_place_in_bin(bin, box_height, box_width, bin_width, bin_height):
                space_left = calculate_space_left(
                    bin, box_height, box_width, bin_width, bin_height
                )
                if space_left < min_space_left:
                    min_space_left = space_left
                    best_bin = bin_num
        if best_bin is None:
            bins.append([(0, 0, box_width, box_height)])
            position.append((len(bins) - 1, 0, 0, 0))
        else:
            if space_left == calculate_space_left(
                bins[best_bin], box_width, box_height, bin_width, bin_height
            ):
                place_box_in_bin(
                    bins[best_bin], box_width, box_height, bin_width, bin_height
                )
                position.append(
                    (best_bin, bins[best_bin][-1][0], bins[best_bin][-1][1], 0)
                )
            else:
                place_box_in_bin(
                    bins[best_bin], box_height, box_width, bin_width, bin_height
                )
                position.append(
                    (best_bin, bins[best_bin][-1][0], bins[best_bin][-1][1], 1)
                )

    initial_positions = [None] * num_boxes
    for i, pos in enumerate(position):
        initial_positions[boxes[i][0]] = pos
    return initial_positions


def can_place_in_bin(bin, w, h, bin_width, bin_height):
    for x in range(bin_width - w + 1):
        for y in range(bin_height - h + 1):
            if all(
                x + w <= bin_x
                or x >= bin_x + bin_w
                or y + h <= bin_y
                or y >= bin_y + bin_h
                for bin_x, bin_y, bin_w, bin_h in bin
            ):
                return True
    return False


def calculate_space_left(bin, w, h, bin_width, bin_height):
    space_left = bin_width * bin_height
    for x, y, bin_w, bin_h in bin:
        space_left -= bin_w * bin_h
    return space_left


def place_box_in_bin(bin, w, h, bin_width, bin_height):
    for x in range(bin_width - w + 1):
        for y in range(bin_height - h + 1):
            if all(
                x + w <= bin_x
                or x >= bin_x + bin_w
                or y + h <= bin_y
                or y >= bin_y + bin_h
                for bin_x, bin_y, bin_w, bin_h in bin
            ):
                bin.append((x, y, w, h))
                return True
    return False


def pso_2d_bin_packing(
    num_boxes, box_dims, bin_width, bin_height, num_particles, max_iter
):
    initial_position = best_fit_decreasing(num_boxes, box_dims, bin_width, bin_height)
    swarm = [
        Particle(num_boxes, bin_width, bin_height, box_dims, initial_position)
        for _ in range(num_particles)
    ]
    global_best_position = None
    global_best_fitness = float("inf")

    for _ in range(max_iter):
        for particle in swarm:
            fitness = fitness_function(
                particle.position, bin_width, bin_height, particle.box_dims
            )
            if fitness < particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = particle.position.copy()
            if fitness < global_best_fitness:
                global_best_fitness = fitness
                global_best_position = particle.position.copy()

        for particle in swarm:
            update_velocity(particle, global_best_position, w=0.5, c1=1.5, c2=1.5)
            update_position(particle)

    # Extract the number of bins used
    bins = {}
    for i, (bin_num, x, y, rotation) in enumerate(global_best_position):
        if bin_num not in bins:
            bins[bin_num] = []
        bins[bin_num].append((i, x, y, rotation))

    # Return the best position, fitness, and the number of bins used
    return global_best_position, global_best_fitness, len(bins), bins


# Example usage
num_boxes = 10
box_dims = [
    (2, 3),
    (4, 5),
    (1, 1),
    (3, 2),
    (2, 2),
    (3, 4),
    (2, 1),
    (4, 4),
    (5, 3),
    (1, 2),
]
bin_width = 10
bin_height = 10
num_particles = 30
max_iter = 100

best_position, best_fitness, num_bins_used, bins = pso_2d_bin_packing(
    num_boxes, box_dims, bin_width, bin_height, num_particles, max_iter
)
print("Best Position:", best_position)
print("Best Fitness:", best_fitness)
print("Number of Bins Used:", num_bins_used)
print("Bins and Box Positions:")
for bin_num, boxes in bins.items():
    print(f"Bin {bin_num}:")
    for box in boxes:
        box_id, x, y, rotation = box
        box_width, box_height = box_dims[box_id]
        if rotation == 1:
            box_width, box_height = box_height, box_width
        center_x = x + box_width / 2
        center_y = y + box_height / 2
        print(
            f"  Box {box_id}: Center ({center_x}, {center_y}), Width {box_width}, Height {box_height}"
        )

# Define a color map for the boxes
colors = plt.cm.get_cmap("tab20", num_boxes)


def plot_bins(bins, box_dims, bin_width, bin_height):
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

    plt.tight_layout()
    plt.show()


# Plot the bins
plot_bins(bins, box_dims, bin_width, bin_height)
