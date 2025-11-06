# stacking_2d_system/slip_layers.py
import os
import glob
import shutil
from math import sqrt
import tempfile
import subprocess
from typing import Tuple, Optional, Dict, List
from gulp_setup import mmanalysis
import numpy as np
from ase.io import read, write
from ase.atoms import Atoms
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

try:
    from  pymatgen.analysis.diffraction.xrd import XRDCalculator
except Exception:
    XRDCalculator = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

adaptor = AseAtomsAdaptor()
# ---------------------- External hooks ----------------------

def optimize_with_gulp(
    ase_atom: Atoms,
    gulp_exe = os.path.expandvars("$HOME/src/gulp-6.0/Src/gulp"), #Optional[str] = None,
    workdir: Optional[str] = None,
    keep_workdir: bool = False,
    timeout: int = 1800,
    add_c_if_2d: bool = True,
    gin_name: str = "job.gin",
    expected_cif: str = "job.cif",
    ) -> Atoms:
    """
    Optimize a structure with GULP and return the optimized ASE Atoms.

    Parameters
    ----------
    ase_atom : Atoms
        Input structure.
    gulp_exe : str, optional
        Path to GULP executable. If None, uses:
          - $GULP_EXE if set, else
          - "gulp" (must be on PATH).
    workdir : str, optional
        Existing directory to use. If None, a temp directory is created.
    keep_workdir : bool
        If True, do not delete the working directory (keep logs/artifacts).
    timeout : int
        Seconds to allow GULP to run before timing out.
    add_c_if_2d : bool
        Forwarded to your mmanalysis.write_gin (adds c lattice if 2D).
    gin_name : str
        Input filename for GULP.
    expected_cif : str
        Preferred output CIF filename that write_gin may emit.

    Returns
    -------
    Atoms
        Optimized structure loaded from the CIF that GULP produced.

    Raises
    ------
    RuntimeError
        If GULP fails or no CIF is produced.
    """
    # Resolve GULP executable
    gulp_exe = gulp_exe or os.environ.get("GULP_EXE") or "gulp"

    # Working directory
    owned_tmp = False
    if workdir is None:
        workdir = tempfile.mkdtemp(prefix="gulp_run_")
        owned_tmp = True
    os.makedirs(workdir, exist_ok=True)

    try:
        gin_path = os.path.join(workdir, gin_name)
        gout_path = os.path.join(workdir, "job.gout")  # stdout log
        got_path = os.path.join(workdir, "job.got")    # traditional GULP text output

        # Build .gin using your utilities
        bonds, mmtypes = mmanalysis.analyze_mm(ase_atom)
        # If your write_gin supports naming the CIF, include it; otherwise it will
        # still produce a default (we handle discovery below).
        try:
            mmanalysis.write_gin(
                gin_path, ase_atom, bonds, mmtypes, add_c_if_2d=add_c_if_2d,
                output_cif=os.path.join(workdir, expected_cif)
            )
        except TypeError:
            # Older signature without output_cif
            mmanalysis.write_gin(gin_path, ase_atom, bonds, mmtypes, add_c_if_2d=add_c_if_2d)

        # Run GULP by piping gin to stdin
        with open(gin_path, "r") as fin, open(gout_path, "w") as fout:
            # No $HOME expansion needed since we resolved gulp_exe already
            proc = subprocess.run(
                [gulp_exe],
                stdin=fin,
                stdout=fout,
                stderr=subprocess.STDOUT,
                cwd=workdir,
                check=False,
                timeout=timeout,
            )

        # Basic failure check: non-zero return code or missing outputs
        if proc.returncode != 0:
            # Capture tail of the log to aid debugging
            tail = ""
            try:
                with open(gout_path, "r") as f:
                    lines = f.readlines()
                    tail = "".join(lines[-200:])
            except Exception:
                pass
            raise RuntimeError(f"GULP failed with code {proc.returncode}.\nLog tail:\n{tail}")

        # Prefer the expected_cif if present
        candidate = os.path.join(workdir, expected_cif)
        if not os.path.isfile(candidate):
            # Otherwise, pick the newest *.cif in workdir
            cifs = glob.glob(os.path.join(workdir, "*.cif"))
            if not cifs:
                # As a fallback, some setups write .res or .cifc; extend as needed
                res_files = glob.glob(os.path.join(workdir, "*.res"))
                if res_files:
                    # If you want, convert .res to Atoms; ASE can often read SHELX .res
                    candidate = max(res_files, key=os.path.getmtime)
                else:
                    # Still nothing — include some log content for debugging
                    tail = ""
                    try:
                        with open(gout_path, "r") as f:
                            lines = f.readlines()
                            tail = "".join(lines[-200:])
                    except Exception:
                        pass
                    raise RuntimeError("GULP did not produce a CIF/RES file.\nLog tail:\n" + tail)
            else:
                candidate = max(cifs, key=os.path.getmtime)

        opt_atoms = read(candidate)

        # Optionally keep artifacts; otherwise clean up temp dir we owned
        if owned_tmp and not keep_workdir:
            shutil.rmtree(workdir, ignore_errors=True)

        return opt_atoms

    except Exception:
        # On exceptions, keep the directory if user requested it
        if owned_tmp and not keep_workdir:
            # If you prefer to keep on error, comment out the next line
            shutil.rmtree(workdir, ignore_errors=True)
        raise

# ---------------------- Utilities ----------------------

def ensure_dir(path: str):
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def xrd_pattern(
    atoms: Atoms,
    two_theta_min: float = 2.0,
    two_theta_max: float = 50.0,
    wavelength: float = 1.5406,  # Cu Kα
    fwhm: float = 0.1,
    points: int = 5000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (2θ, intensity) on a dense grid by Gaussian broadening of sticks.
    Intensities are normalized to max=1.
    """
    if XRDCalculator is None:
        raise ImportError(
            "pytmagen XRDCalculator not available. Please install ASE with xrd support."
        )
    structure = structure = adaptor.get_structure(atoms)
    calc = XRDCalculator(wavelength=wavelength)
    sticks = calc.get_pattern(structure, two_theta_range=(two_theta_min, two_theta_max))
    tth_sticks = np.asarray(sticks.x, dtype=float)
    I_sticks = np.asarray(sticks.y, dtype=float)

    grid = np.linspace(two_theta_min, two_theta_max, points)
    inten = np.zeros_like(grid)
    if len(tth_sticks) > 0:
        sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        for t, I in zip(tth_sticks, I_sticks):
            inten += I * np.exp(-0.5 * ((grid - t) / sigma) ** 2)

    if inten.max() > 0:
        inten = inten / inten.max()
    return grid, inten


def resample_to_step(two_theta: np.ndarray, intensity: np.ndarray, step: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resamples (2θ, I) to a uniform grid with the given step (default 0.2°).
    Uses linear interpolation; extrapolation is clamped to 0.
    """
    two_theta = np.asarray(two_theta, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    tmin, tmax = float(two_theta.min()), float(two_theta.max())
    grid = np.round(np.arange(tmin, tmax + 1e-9, step) / step) * step  # exact multiples
    if len(two_theta) < 2:
        return grid, np.zeros_like(grid)
    I = np.interp(grid, two_theta, intensity, left=0.0, right=0.0)
    return grid, I


def ensure_exp_step_0p2(two_theta: np.ndarray, intensity: np.ndarray, tol: float = 0.02) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Ensures experimental data is on a 0.2° step grid. If not within tolerance,
    resamples to 0.2°. Returns (tth_grid, I_norm, used_step).
    """
    two_theta = np.asarray(two_theta, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    diffs = np.diff(two_theta)
    median_step = float(np.median(diffs)) if len(diffs) else 0.2
    if abs(median_step - 0.2) > tol:
        # resample to 0.2
        tth, I = resample_to_step(two_theta, intensity, step=0.2)
    else:
        tth, I = two_theta.copy(), intensity.copy()
    # normalize experimental intensities to max=1
    if I.max() > 0:
        I = I / I.max()
    return tth, I, 0.2


def first_peak(two_theta: np.ndarray, intensity: np.ndarray, min_rel_height: float = 0.05) -> Optional[float]:
    """
    Smallest-2θ peak above a relative threshold, else global max location.
    """
    if len(two_theta) == 0 or len(intensity) == 0:
        return None
    I = intensity / (intensity.max() if intensity.max() > 0 else 1.0)
    mask = I >= min_rel_height
    if not np.any(mask):
        return float(two_theta[np.argmax(I)])
    idxs = np.where(mask)[0]
    return float(two_theta[idxs[0]])


def scale_least_squares(y_obs: np.ndarray, y_calc: np.ndarray) -> float:
    """
    Optimal scale factor s minimizing || y_obs - s*y_calc ||_2^2.
    """
    denom = float(np.dot(y_calc, y_calc))
    if denom == 0:
        return 0.0
    return float(np.dot(y_obs, y_calc) / denom)


def compute_r_factors(y_obs: np.ndarray, y_calc: np.ndarray, weights: Optional[np.ndarray] = None, P: int = 0) -> Dict[str, float]:
    """
    Conventional powder profile R-factors with unit weights if none provided.
      Rp   = sum |y_o - y_c| / sum |y_o|
      Rwp  = sqrt( sum w (y_o - y_c)^2 / sum w y_o^2 )
      Rexp = sqrt( (N - P) / sum w y_o^2 )   [P = number of refined parameters; use 0 here]
      GoF  = (Rwp/Rexp)^2
    """
    y_obs = np.asarray(y_obs, dtype=float)
    y_calc = np.asarray(y_calc, dtype=float)
    N = y_obs.size
    if weights is None:
        w = np.ones_like(y_obs)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != y_obs.shape:
            w = np.ones_like(y_obs)

    # avoid negatives from background quirks
    y_obs = np.clip(y_obs, 0.0, None)
    y_calc = np.clip(y_calc, 0.0, None)

    denom_abs = np.sum(np.abs(y_obs)) if np.sum(np.abs(y_obs)) > 0 else 1.0
    Rp = float(np.sum(np.abs(y_obs - y_calc)) / denom_abs)

    denom_w = float(np.sum(w * y_obs * y_obs)) if np.sum(w * y_obs * y_obs) > 0 else 1.0
    Rwp = float(np.sqrt(np.sum(w * (y_obs - y_calc) ** 2) / denom_w))

    # Use P=0 unless you account for refined parameters separately
    Rexp = float(np.sqrt(max(N - P, 1) / denom_w))
    GoF = float((Rwp / Rexp) ** 2) if Rexp > 0 else np.inf

    return dict(Rp=Rp, Rwp=Rwp, Rexp=Rexp, GoF=GoF)


def peak_window_mask(two_theta: np.ndarray, center: float, half_width: float = 0.3) -> np.ndarray:
    return (two_theta >= center - half_width) & (two_theta <= center + half_width)


def plot_overlay_pxrd(exp_tth, exp_I, sim_tth, sim_I, out_png: str, title: str = "PXRD Comparison"):
    plt.figure(figsize=(7, 4.5))
    plt.plot(exp_tth, exp_I, label="Experimental")
    plt.plot(sim_tth, sim_I, label="Computed")
    plt.xlabel(r"2$\theta$ (°)")
    plt.ylabel("Normalized intensity (a.u.)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# ---------------------- Main class ----------------------

class CreateStack:
    """
    Build AA/AB + systematic slips, simulate PXRD, enforce 0.2° grid
    and evaluate both first-peak match and full-pattern residuals.
    """

    def __init__(self, filename: str, interlayer_dist: float = 4.0, output_dir: str = "."):
        self.filename = filename
        self.interlayer_dist = interlayer_dist
        self.output_dir = output_dir
        ensure_dir(output_dir)

        self.base_name = os.path.basename(filename).split(".")[0]
        self.monolayer = read(filename).copy()
        cell = self.monolayer.get_cell().copy()
        cell[2, 2] = interlayer_dist
        self.monolayer.set_cell(cell, scale_atoms=False)
        self.bilayer = self.monolayer * (1, 1, 2)

        x, y, z = self.monolayer.cell.lengths().tolist()
        self.x, self.y, self.z = x, y, z

    # ----- geometry builders -----

    def _make_shifted(self, dx: float, dy: float, dz: float = 0.0) -> Atoms:
        atoms = self.bilayer.copy()
        n = len(self.monolayer)
        pos = atoms.get_positions()
        pos[n:, 0] += dx
        pos[n:, 1] += dy
        pos[n:, 2] += dz
        atoms.set_positions(pos)
        return atoms

    def create_aa(self) -> Atoms:
        return self.bilayer.copy()

    def create_ab(self) -> Atoms:
        a_vec = self.monolayer.cell[0]
        b_vec = self.monolayer.cell[1]
        cell_a = float(np.linalg.norm(a_vec))
        cell_b = float(np.linalg.norm(b_vec))
        dx = cell_a / 2.0
        dy = (cell_b / 6.0) * sqrt(3.0)
        return self._make_shifted(dx, dy, 0.0)

    # ----- IO helpers -----

    def _stack_name(self, kind: str, shift: Tuple[float, float, float]) -> str:
        if kind in ("aa", "ab"):
            return kind
        dx, dy, _ = shift
        if kind == "x":
            return f"x_{dx:.1f}"
        if kind == "y":
            return f"y_{dy:.1f}"
        if kind == "xy":
            return f"xy_{dx:.1f}"
        return f"shift_{dx:.2f}_{dy:.2f}"

    def _write_cif(self, atoms: Atoms, tag: str) -> str:
        cif_path = os.path.join(self.output_dir, f"{self.base_name}_{tag}.cif")
        write(cif_path, atoms)
        return cif_path

    def _write_report_txt(
        self,
        atoms: Atoms,
        tag: str,
        report_lines: List[str],
    ) -> str:

        txt_path = os.path.join(self.output_dir, f"{self.base_name}.txt")

        lines = []
        lines.append("=" * 72)
        lines.append(f" Calculation Report for: {self.base_name}_{tag}")
        lines.append("=" * 72)
        lines.append("")

        lines.append(f" System Name     : {self.base_name}")
        lines.append(f" Stacking : {tag}")
        lines.append(f" Number of Atoms : {len(atoms)}")
        lines.append("")


        lines.append(f"{'Atom':<6}{'X':>14}{'Y':>14}{'Z':>14}")
        lines.append("-" * 48)
        for sym, (x, y, z) in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
            lines.append(f"{sym:<6}{x:14.8f}{y:14.8f}{z:14.8f}")
        lines.append("End\n")

        lines.append("Lattice")
        lines.append(f"{'a':<3} {atoms.cell[0,0]:12.6f} {atoms.cell[0,1]:12.6f} {atoms.cell[0,2]:12.6f}")
        lines.append(f"{'b':<3} {atoms.cell[1,0]:12.6f} {atoms.cell[1,1]:12.6f} {atoms.cell[1,2]:12.6f}")
        lines.append(f"{'c':<3} {atoms.cell[2,0]:12.6f} {atoms.cell[2,1]:12.6f} {atoms.cell[2,2]:12.6f}")
        lines.append("End\n")

        #
        if report_lines:
            lines.append("PXRD and Similarity Report")
            for line in report_lines:
                lines.append(f"  {line}")
            lines.append("")
            lines.append(f" End of Report for {self.base_name}\n")

        with open(txt_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return txt_path


    # ----- matching core -----

    def _simulate_and_match(
        self,
        atoms_in: Atoms,
        tag: str,
        exp_tth: np.ndarray,
        exp_I: np.ndarray,
        optimize: bool,
        tol_first_peak_deg: float,
        sim_cfg: Dict,
        zero_shift_range: float = 0.2,
        zero_shift_step: float = 0.01,
        first_peak_window: float = 0.3,
    ) -> Dict:
        """
        Simulate PXRD, align to exp (0.2° grid), test small zero-shift,
        scale analytically, compute conventional residuals and first-peak stats.
        """
        # cif_in = self._write_cif(atoms_in, tag)
        atoms = optimize_with_gulp(atoms_in) if optimize else atoms_in.copy()
        # cif_used = self._write_cif(atoms, f"{tag}_final")
        cif_used = os.path.join(self.output_dir, f"{self.base_name}_{tag}.cif")


        # Simulate on dense grid then resample to 0.2°
        sim_tth_raw, sim_I_raw = xrd_pattern(
            atoms,
            two_theta_min=sim_cfg.get("tth_min", np.min(exp_tth)),
            two_theta_max=sim_cfg.get("tth_max", np.max(exp_tth)),
            wavelength=sim_cfg.get("wavelength", 1.5406),
            fwhm=sim_cfg.get("fwhm", 0.1),
            points=sim_cfg.get("points", 5000),
        )

        # Ensure both on the same 0.2° grid as experimental
        # (exp_tth, exp_I) are already 0.2° normalized at callsite
        sim_on_exp = np.interp(exp_tth, sim_tth_raw, sim_I_raw, left=0.0, right=0.0)

        # zero-shift scan (professional quick alignment when not refining)
        shifts = np.arange(-zero_shift_range, zero_shift_range + 1e-12, zero_shift_step)
        best = {
            "Rwp": np.inf,
            "shift": 0.0,
            "scale": 1.0,
            "ycalc": sim_on_exp.copy(),
        }
        for s in shifts:
            sim_shifted = np.interp(exp_tth, sim_tth_raw + s, sim_I_raw, left=0.0, right=0.0)
            # optimal scale
            scale = scale_least_squares(exp_I, sim_shifted)
            ycalc = np.clip(scale * sim_shifted, 0.0, None)
            r = compute_r_factors(exp_I, ycalc, weights=None, P=0)
            if r["Rwp"] < best["Rwp"]:
                best.update(shift=float(s), scale=float(scale), ycalc=ycalc, **r)

        # full-pattern scores
        ycalc = best["ycalc"]
        r_full = compute_r_factors(exp_I, ycalc, weights=None, P=0)
        cos_sim = float(np.dot(exp_I, ycalc) / (np.linalg.norm(exp_I) * np.linalg.norm(ycalc) + 1e-12))
        pear_r = pearson_r(exp_I, ycalc)

        # first-peak stats
        exp_first = first_peak(exp_tth, exp_I)
        sim_first = first_peak(exp_tth, ycalc)  # after shift/scale on same grid
        d_first = abs(exp_first - sim_first) if (exp_first is not None and sim_first is not None) else None

        # peak-window R (local Rp) around the first experimental peak
        Rp_peak = None
        if exp_first is not None:
            mask = peak_window_mask(exp_tth, exp_first, half_width=first_peak_window)
            if np.any(mask):
                Rp_peak = compute_r_factors(exp_I[mask], ycalc[mask])["Rp"]

        # Build report content
        report = []
        report.append("Report")
        report.append(f"Stacking: {tag}")
        # report.append(f"CIF: {os.path.basename(cif_used)}")
        report.append(f"Zero shift applied (deg): {best['shift']:.3f}")
        report.append(f"Scale factor: {best['scale']:.6f}")
        if exp_first is not None:
            report.append(f"Experimental first peak (2θ): {exp_first:.4f} °")
        if sim_first is not None:
            report.append(f"Computed first peak  (2θ): {sim_first:.4f} °")
        if d_first is not None:
            report.append(f"|Δ first peak|: {d_first:.4f} °")
        report.append(f"Rp (profile): {r_full['Rp']:.4f}")
        report.append(f"Rwp (weighted): {r_full['Rwp']:.4f}")
        report.append(f"Rexp: {r_full['Rexp']:.4f}")
        report.append(f"GoF (chi^2): {r_full['GoF']:.4f}")
        if Rp_peak is not None:
            report.append(f"R-factor (first-peak window ±0.3°): {Rp_peak:.4f}")
        report.append(f"Cosine similarity: {cos_sim:.4f}")
        report.append(f"Pearson r: {pear_r:.4f}")
        report.append("")
        report.append("Intensity 2theta (computed, normalized, 0.2° grid)")
        for t, I in zip(exp_tth, ycalc):
            report.append(f"{I:.6f} {t:.4f}")

        # Write report file
        txt_path = self._write_report_txt(atoms, tag, report)

        # Check match by first peak tolerance
        is_match = (d_first is not None) and (d_first <= tol_first_peak_deg)

        plot_path = None
        if is_match:
            plot_path = os.path.join(self.output_dir, f"{self.base_name}_{tag}_pxrd_match.png")
            plot_overlay_pxrd(exp_tth, exp_I, exp_tth, ycalc, plot_path, title=f"PXRD: {self.base_name} ({tag})")
            with open(txt_path, "a", encoding="utf-8") as f:
                f.write("\n\nMatch found\n")
            # self._write_cif(atoms, f"{tag}_final")

        return dict(
            tag=tag,
            cif=cif_used,
            txt=txt_path,
            match=is_match,
            first_peak_diff=d_first,
            Rp=r_full["Rp"],
            Rwp=r_full["Rwp"],
            Rexp=r_full["Rexp"],
            GoF=r_full["GoF"],
            Rp_first_peak=Rp_peak,
            cosine=cos_sim,
            pearson=pear_r,
            zero_shift=best["shift"],
            scale=best["scale"],
            plot=plot_path,
        )

    # ---------------------- public API ----------------------

    def search_best_stacking(
        self,
        exp_two_theta: np.ndarray,
        exp_intensity: np.ndarray,
        optimize: bool = False,
        tol_first_peak_deg: float = 0.05,
        slip_step: float = 0.5,
        slip_max: float = 8.0,
        sim_cfg: Optional[Dict] = None,
        rwp_threshold: Optional[float] = None,   # e.g., 0.10 for 10% if you want a full-pattern stop
    ) -> Dict:
        """
        1) Enforce experimental grid to 0.2°
        2) Try AA, AB, then slips (x, y, xy) at 0.5 Å up to 8 Å.
        3) For each: simulate, zero-shift & scale against experimental 0.2° grid,
           compute residuals & first-peak match; stop if |Δ2θ| ≤ tol.
           Optionally also stop if Rwp ≤ rwp_threshold.
        Returns the first structure that meets criteria; otherwise the last tried.
        """
        sim_cfg = sim_cfg or {}

        # Ensure experimental step = 0.2°
        exp_tth, exp_I, _ = ensure_exp_step_0p2(exp_two_theta, exp_intensity, tol=0.02)

        # AA
        res_aa = self._simulate_and_match(
            self.create_aa(), "aa", exp_tth, exp_I, optimize, tol_first_peak_deg, sim_cfg
        )
        if res_aa["match"] or (rwp_threshold is not None and res_aa["Rwp"] <= rwp_threshold):
            return res_aa

        # AB
        res_ab = self._simulate_and_match(
            self.create_ab(), "ab", exp_tth, exp_I, optimize, tol_first_peak_deg, sim_cfg
        )
        if res_ab["match"] or (rwp_threshold is not None and res_ab["Rwp"] <= rwp_threshold):
            return res_ab

        # Slips
        shifts = np.arange(slip_step, slip_max + 1e-9, slip_step)
        last_res = res_ab
        for s in shifts:
            # x
            res_x = self._simulate_and_match(
                self._make_shifted(s, 0.0), f"x_{s:.1f}", exp_tth, exp_I, optimize, tol_first_peak_deg, sim_cfg
            )
            last_res = res_x
            if res_x["match"] or (rwp_threshold is not None and res_x["Rwp"] <= rwp_threshold):
                return res_x

            # y
            res_y = self._simulate_and_match(
                self._make_shifted(0.0, s), f"y_{s:.1f}", exp_tth, exp_I, optimize, tol_first_peak_deg, sim_cfg
            )
            last_res = res_y
            if res_y["match"] or (rwp_threshold is not None and res_y["Rwp"] <= rwp_threshold):
                return res_y

            # xy
            res_xy = self._simulate_and_match(
                self._make_shifted(s, s), f"xy_{s:.1f}", exp_tth, exp_I, optimize, tol_first_peak_deg, sim_cfg
            )
            last_res = res_xy
            if res_xy["match"] or (rwp_threshold is not None and res_xy["Rwp"] <= rwp_threshold):
                return res_xy

        # If nothing satisfied stopping criteria, return the last tried.
        return last_res


# ---------------------- Extra stats ----------------------

def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return 0.0
    a = a.astype(float)
    b = b.astype(float)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
