#!/usr/bin/env python3
import argparse
import numpy as np
from stacking_2d_systems.auto_stack import CreateStack
from stacking_2d_systems.file_typer import load_data


def main():
    parser = argparse.ArgumentParser(
        description="Auto-stacking PXRD matcher using stacking_2d_systems."
    )

    parser.add_argument(
        "--input", "-i", required=True, help="Input PXRD file (CSV, XLSX, etc.)"
    )
    parser.add_argument(
        "--cif", "-c", required=True, help="CIF file of the monolayer structure"
    )
    parser.add_argument(
        "--interlayer", "-d", type=float, default=4.0,
        help="Interlayer distance (default: 4.0 Å)"
    )
    parser.add_argument(
        "--outdir", "-o", default="pxrd_scan",
        help="Output directory (default: pxrd_scan)"
    )
    parser.add_argument(
        "--optimize", action="store_true",
        help="Enable GULP optimization"
    )
    parser.add_argument(
        "--slipstep", type=float, default=0.5,
        help="Slip step size (default: 0.5 Å)"
    )
    parser.add_argument(
        "--slipmax", type=float, default=8.0,
        help="Maximum slip (default: 8 Å)"
    )
    parser.add_argument(
        "--fwhm", type=float, default=0.10,
        help="PXRD FWHM for simulation (default: 0.10)"
    )
    parser.add_argument(
        "--rwp", type=float, default=None,
        help="Rwp threshold for stopping early (optional)"
    )

    args = parser.parse_args()

    # -----------------------
    # Load PXRD experimental data
    # -----------------------
    pxrd = load_data(args.input)
    exp_two_theta = pxrd.get("two_theta").values
    exp_intensity = pxrd.get("intensity").values

    print("Loaded PXRD:")
    print("2θ:", exp_two_theta)
    print("Intensity:", exp_intensity)

    # -----------------------
    # Create stacking model
    # -----------------------
    cs = CreateStack(
        args.cif,
        interlayer_dist=args.interlayer,
        output_dir=args.outdir,
    )

    # -----------------------
    # Run stacking search
    # -----------------------
    sim_cfg = dict(
        tth_min=float(np.min(exp_two_theta)),
        tth_max=float(np.max(exp_two_theta)),
        wavelength=1.5406,
        fwhm=args.fwhm,
        points=5000,
    )

    result = cs.search_best_stacking(
        exp_tth=exp_two_theta,
        exp_I=exp_intensity,
        optimize=args.optimize,
        slip_step=args.slipstep,
        slip_max=args.slipmax,
        tol_first_peak_deg=0.05,
        sim_cfg=sim_cfg,
        rwp_threshold=args.rwp,
    )

    print("\n===== Best Stacking Result =====")
    print(result)


if __name__ == "__main__":
    main()
