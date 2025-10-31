# pyproject/cli.py

import argparse
import os
from stacking_2d_systems.slip_layers import CreateStack


def main():
    parser = argparse.ArgumentParser(
        description="Generate different stacking configurations of 2D materials."
    )

    parser.add_argument(
        "filename",
        type=str,
        help="Path to the input monolayer structure file (e.g., .cif)."
    )
    parser.add_argument(
        "--interlayer-dist",
        type=float,
        default=4.2,
        help="Distance between layers in the bilayer structure. Default is 3.2 Angstroms."
    )
    parser.add_argument(
        "--stacking",
        choices=["ab", "aa", "x", "y", "xy", "all"],
        type=str,
        default="all",
        help="Type of stacking: 'ab', 'aa', 'x', 'y', 'xy', or 'all'. Default is 'all'."
    )
    parser.add_argument(
        "--max-length",
        type=float,
        help="Maximum translation length along the specified axis (x, y, or xy)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to save the output CIF files. Default is the current directory."
    )

    args = parser.parse_args()

    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Create the stack object with the specified output directory
    stack = CreateStack(
        args.filename,
        interlayer_dist=args.interlayer_dist,
        output_dir=args.output_dir
    )

    # Generate the specified stacking
    if args.stacking == "ab":
        stack.create_ab_stacking()
        print(f"AB stacking created: {os.path.join(args.output_dir, stack.base_name)}_ab.cif")
    elif args.stacking == "aa":
        stack.create_aa_stacking()
        print(f"AA stacking created: {os.path.join(args.output_dir, stack.base_name)}_aa.cif")
    elif args.stacking == "x":
        stack.stack_along_x(args.max_length)
        print(f"Stacking along x created with max length {args.max_length or stack.x / 2.0} Angstroms")
    elif args.stacking == "y":
        stack.stack_along_y(args.max_length)
        print(f"Stacking along y created with max length {args.max_length or stack.y / 2.0} Angstroms")
    elif args.stacking == "xy":
        stack.stack_along_xy(args.max_length)
        print(f"Stacking along xy created with max length {args.max_length or (stack.x + stack.y) / 2.0} Angstroms")
    elif args.stacking == "all":
        stack.create_ab_stacking()
        stack.create_aa_stacking()
        stack.stack_along_x(args.max_length)
        stack.stack_along_y(args.max_length)
        stack.stack_along_xy(args.max_length)
        print("All stacking configurations created.")


if __name__ == "__main__":
    main()
