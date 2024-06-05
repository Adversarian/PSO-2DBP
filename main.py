from pprint import pformat

from utils import (
    get_arg_parser,
    parse_test_case_from_file,
    plot_bins,
    pso_2d_bin_packing,
)

if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()
    print(f"Initializing with the following args:\n{pformat(vars(args))}")
    num_boxes, box_dims, bin_width, bin_height = parse_test_case_from_file(
        path=args.test_case
    )
    best_position, best_fitness, num_bins, bins = pso_2d_bin_packing(
        num_boxes=num_boxes,
        box_dims=box_dims,
        bin_width=bin_width,
        bin_height=bin_height,
        num_particles=args.num_particles,
        max_iter=args.max_iter,
        w=args.w,
        c1=args.c1,
        c2=args.c2,
        time_limit=args.time_limit,
        verbose=args.verbose,
    )
    plot_bins(
        bins=bins,
        box_dims=box_dims,
        bin_width=bin_width,
        bin_height=bin_height,
        fitness_value=best_fitness,
        save_to_file=args.save_fig,
    )
