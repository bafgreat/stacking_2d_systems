#!/usr/bin/env python3
import argparse
import numpy as np
from ase.io import read
from stacking_2d_systems.auto_stack import CreateStack
from stacking_2d_systems.file_typer import load_data
from stacking_2d_systems import auto_stack


def run_simulation(cif_file, asc_file, out_png, out_csv,
                   wavelength=None, fwhm=None, tth_min=None, tth_max=None):

    # Load structure
    atoms = read(cif_file)

    # Load experimental PXRD
    pxrd = load_data(asc_file)
    exp_two_theta = pxrd.get('two_theta').values
    exp_intensity = pxrd.get('intensity').values

    # Simulation configuration
    sim_cfg = {
        "wavelength": wavelength if wavelength else 1.5406,
        "fwhm": fwhm if fwhm else 0.12,
        "tth_min": tth_min if tth_min else float(np.min(exp_two_theta)),
        "tth_max": tth_max if tth_max else float(np.max(exp_two_theta)),
    }

    # Perform the simulation
    res = auto_stack.simulate_and_plot_pxrd_from_exp(
        atoms,
        exp_two_theta,
        exp_intensity,
        out_png=out_png,
        sim_cfg=sim_cfg,
        save_csv=out_csv,
    )

    print(f"Rwp: {res['Rwp']}")
    print(f"Plot saved: {res['png']}")
    return res


def main():
    parser = argparse.ArgumentParser(
        description="Simulate PXRD from CIF and experimental ASC file using stacking_2d_systems."
    )

    parser.add_argument("-c", "--cif", required=True,
                        help="Path to CIF structure (e.g., CC1a.cif)")
    parser.add_argument("-p", "--pxrd", required=True,
                        help="Path to experimental PXRD ASC file (e.g., CC1.ASC)")
    parser.add_argument("-o", "--out", required=True,
                        help="Output PNG file path (e.g., PXRDs/output.png)")
    parser.add_argument("--csv", default="overlay_data.csv",
                        help="Output CSV for overlay data (default: overlay_data.csv)")

    # Optional simulation parameters
    parser.add_argument("--wavelength", type=float,
                        help="Wavelength for PXRD simulation (default: Cu Kα 1.5406 Å)")
    parser.add_argument("--fwhm", type=float,
                        help="FWHM broadening (default: 0.12)")
    parser.add_argument("--tth-min", type=float,
                        help="Override minimum 2θ (default: from experiment)")
    parser.add_argument("--tth-max", type=float,
                        help="Override maximum 2θ (default: from experiment)")

    args = parser.parse_args()

    run_simulation(
        cif_file=args.cif,
        asc_file=args.pxrd,
        out_png=args.out,
        out_csv=args.csv,
        wavelength=args.wavelength,
        fwhm=args.fwhm,
        tth_min=args.tth_min,
        tth_max=args.tth_max,
    )
