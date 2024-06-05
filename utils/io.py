import argparse


def parse_test_case_from_file(path):
    with open(path, "r") as test_case:
        lines = test_case.readlines()
        num_boxes = int(lines[0].strip())
        bin_width, bin_height = [int(num) for num in lines[1].strip().split()]
        box_dims = []
        for i in range(num_boxes):
            box_dims.append([int(num) for num in lines[i + 2].strip().split()])
    return num_boxes, box_dims, bin_width, bin_height


def get_arg_parser():
    parser = argparse.ArgumentParser(
        description="A solver based on PSO for the 2D bin-packing problem"
    )
    parser.add_argument(
        "--test_case", "-tc", type=str, help="Path to the test case file."
    )
    parser.add_argument(
        "--max_iter",
        "-mi",
        type=int,
        default=100,
        help="Maximum number of iterations allowed for the PSO algorithm. Defaults to `100`.",
    )
    parser.add_argument(
        "--num_particles",
        "-np",
        type=int,
        default=100,
        help="Number of particles to initialize the swarm with. Defaults to `100`.",
    )
    parser.add_argument(
        "-w",
        type=float,
        default=0.5,
        help="Inertia weight. Defaults to `0.5`.",
    )
    parser.add_argument(
        "-c1",
        type=float,
        default=1.5,
        help="Cognitive coefficient. Defaults to `1.5`.",
    )
    parser.add_argument(
        "-c2",
        type=float,
        default=2.0,
        help="Social coefficient. Defaults to `2.0`.",
    )
    parser.add_argument(
        "--time_limit",
        "-t",
        type=float,
        default=None,
        help="Time limit for the PSO algorithm. Defaults to `None`.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print the solution after the algorithm finishes its run.",
    )
    parser.add_argument(
        "--save_fig",
        "-s",
        action="store_true",
        help="Whether or not to save a figure of the final solution in a file.",
    )
    return parser
