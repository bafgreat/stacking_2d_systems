import argparse
import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from ase.io import read

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.diffraction.xrd import XRDCalculator

from stacking_2d_systems.file_typer import load_data
import pandas as pd


def gaussian_broaden(two_theta_peaks, intensities, grid, fwhm=0.1):
    """
    Turn discrete XRD peaks into a continuous profile using Gaussian broadening.
    """
    grid = np.asarray(grid)
    profile = np.zeros_like(grid, dtype=float)
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))  # FWHM → σ

    for tt, I in zip(two_theta_peaks, intensities):
        profile += I * np.exp(-0.5 * ((grid - tt) / sigma) ** 2)

    return profile


def plot_pxrd_ase_vs_experimental(
    atoms: Atoms,
    two_theta_exp: np.ndarray,
    intensity_exp: np.ndarray,
    wavelength: float = 1.5406,  # Cu Kα (Å)
    fwhm_sim: float = 0.12,
    n_points: int = 5000,
    tt_min: float | None = None,
    tt_max: float | None = None,
    savefig: str | None = None,
    csv_path: str | None = None,
):
    """
    Compute and plot simulated PXRD from an ASE Atoms object together with
    an experimental PXRD pattern.
    """

    two_theta_exp = np.asarray(two_theta_exp, dtype=float)
    intensity_exp = np.asarray(intensity_exp, dtype=float)

    # --- 1. ASE → pymatgen Structure ---
    structure = AseAtomsAdaptor.get_structure(atoms)

    # --- 2. Determine 2θ range from experimental data (unless overridden) ---
    if tt_min is None:
        tt_min = float(two_theta_exp.min())
    if tt_max is None:
        tt_max = float(two_theta_exp.max())

    # --- 3. Compute simulated PXRD using pymatgen ---
    calc = XRDCalculator(wavelength=wavelength)
    pattern = calc.get_pattern(structure, two_theta_range=(tt_min, tt_max))

    two_theta_peaks = np.array(pattern.x)  # 2θ peak positions
    intensities_peaks = np.array(pattern.y)

    # Build a dense grid and broaden to get a smooth curve
    two_theta_grid = np.linspace(tt_min, tt_max, n_points)
    intensity_sim = gaussian_broaden(
        two_theta_peaks,
        intensities_peaks,
        two_theta_grid,
        fwhm=fwhm_sim,
    )

    # --- 4. Normalize & vertically offset for pretty comparison ---
    intensity_exp_norm = intensity_exp / intensity_exp.max()
    intensity_sim_norm = intensity_sim / intensity_sim.max()

    offset = 1.2  # vertical offset between simulated and experimental

    # --- 5. Plot in a style similar to your example ---
    fig, ax = plt.subplots(figsize=(9, 6), dpi=400)

    # Simulated (bottom, black)
    ax.plot(two_theta_grid, intensity_sim_norm, lw=1.5, label="simulated", color="k")

    # Experimental (top, blue, shifted up)
    ax.plot(two_theta_exp, intensity_exp_norm + offset, lw=1.2, label="synthesised", color="b")

    # Cosmetics
    ax.set_xlim(tt_min, tt_max)
    ax.set_xlabel(r"2$\theta$ (degree)", fontsize=20, fontweight="bold")
    ax.set_ylabel("Intensity (a.u.)", fontsize=20, fontweight="bold")

    ax.tick_params(axis="both", labelsize=20)
    ax.set_yticks([])  # cleaner like your reference figure

    # Legend
    ax.legend(loc="upper right", fontsize=18)

    # Thicker box edges like typical PXRD figures
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    fig.tight_layout()

    # --- 6. Save figure and overlay data if requested ---
    if savefig is not None:
        fig.savefig(savefig, bbox_inches="tight")

    if csv_path is not None:
        # Interpolate experimental onto the simulated grid for a neat overlay CSV
        exp_interp = np.interp(two_theta_grid, two_theta_exp, intensity_exp_norm)
        df = pd.DataFrame(
            {
                "two_theta": two_theta_grid,
                "I_exp_norm_shifted": exp_interp + offset,
                "I_sim_norm": intensity_sim_norm,
            }
        )
        df.to_csv(csv_path, index=False)

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Simulate PXRD from CIF and experimental ASC file using stacking_2d_systems."
    )

    parser.add_argument("-c", "--cif", required=True,
                        help="Path to CIF structure (e.g., CC1a.cif)")
    parser.add_argument("-p", "--pxrd", required=True,
                        help="Path to experimental PXRD ASC file (e.g., CC1.ASC)")
    parser.add_argument("-o", "--out", required=True,
                        help="Output PNG file path (e.g., pxrd_comparison.png)")
    parser.add_argument("--csv", default="overlay_data.csv",
                        help="Output CSV for overlay data (default: overlay_data.csv)")

    # Optional simulation parameters
    parser.add_argument("--wavelength", type=float, default=1.5406,
                        help="Wavelength for PXRD simulation in Å (default: Cu Kα 1.5406 Å)")
    parser.add_argument("--fwhm", type=float, default=0.12,
                        help="FWHM broadening in degrees (default: 0.12)")
    parser.add_argument("--tth-min", type=float,
                        help="Override minimum 2θ in degrees (default: from experiment)")
    parser.add_argument("--tth-max", type=float,
                        help="Override maximum 2θ in degrees (default: from experiment)")

    args = parser.parse_args()

    atoms = read(args.cif)

    pxrd = load_data(args.pxrd)  # expects DataFrame with 'two_theta' and 'intensity'
    two_theta_exp = pxrd.get("two_theta").values
    intensity_exp = pxrd.get("intensity").values

    plot_pxrd_ase_vs_experimental(
        atoms,
        two_theta_exp,
        intensity_exp,
        wavelength=args.wavelength,
        fwhm_sim=args.fwhm,
        tt_min=args.tth_min,
        tt_max=args.tth_max,
        savefig=args.out,
        csv_path=args.csv,
    )


if __name__ == "__main__":
    main()
