from time import perf_counter

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

    penalty_multiplier = 1000  # High penalty to prioritize feasible solutions

    return num_bins_used + penalty * penalty_multiplier


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
                space_left = calculate_space_left(bin, bin_width, bin_height)
                if space_left < min_space_left:
                    min_space_left = space_left
                    best_bin = bin_num
            if can_place_in_bin(bin, box_height, box_width, bin_width, bin_height):
                space_left = calculate_space_left(bin, bin_width, bin_height)
                if space_left < min_space_left:
                    min_space_left = space_left
                    best_bin = bin_num
        if best_bin is None:
            bins.append([(0, 0, box_width, box_height)])
            position.append((len(bins) - 1, 0, 0, 0))
        else:
            if space_left == calculate_space_left(
                bins[best_bin], bin_width, bin_height
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


def calculate_space_left(bin, bin_width, bin_height):
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
    num_boxes,
    box_dims,
    bin_width,
    bin_height,
    num_particles,
    max_iter,
    w=0.5,
    c1=1.5,
    c2=1.5,
    time_limit=None,
    verbose=True,
):
    if time_limit is not None:
        tic = perf_counter()
    initial_position = best_fit_decreasing(num_boxes, box_dims, bin_width, bin_height)
    swarm = [
        Particle(num_boxes, bin_width, bin_height, box_dims, initial_position)
        for _ in range(num_particles)
    ]
    global_best_position = None
    global_best_fitness = float("inf")

    for i in range(max_iter):
        if time_limit is not None:
            toc = perf_counter()
            if toc - tic >= time_limit:
                print(f"Time limit exhausted. Runtime: {toc - tic}")
                break
            else:
                tic = perf_counter()
        if global_best_fitness < 1000:
            print(f"Converged in {i} iterations.")
            break
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
            update_velocity(particle, global_best_position, w, c1, c2)
            update_position(particle)

    # Extract the number of bins used
    bins = {}
    for i, (bin_num, x, y, rotation) in enumerate(global_best_position):
        if bin_num not in bins:
            bins[bin_num] = []
        bins[bin_num].append((i, x, y, rotation))

    if verbose:
        print("Best Position:", global_best_position)
        print("Best Fitness:", global_best_fitness)
        print("Number of Bins Used:", len(bins))
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
    # Return the best position, fitness, and the number of bins used
    return global_best_position, global_best_fitness, len(bins), bins
