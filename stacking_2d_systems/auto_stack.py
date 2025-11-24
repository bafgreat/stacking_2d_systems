# stacking_2d_system/slip_layers.py
import os
import re
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
from mofstructure import mofdeconstructor

try:
    from  pymatgen.analysis.diffraction.xrd import XRDCalculator
except Exception:
    XRDCalculator = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

adaptor = AseAtomsAdaptor()
# ---------------------- External hooks ----------------------

def create_layer(ase_atom, interlayer_dist):
    atom_neighbors, _ = mofdeconstructor.compute_ase_neighbour(ase_atom)
    fragments = mofdeconstructor.connected_components(atom_neighbors)
    if len(fragments) == 1:
        cell = ase_atom.get_cell().copy()
        cell[2, 2] = interlayer_dist
        ase_atom.set_cell(cell, scale_atoms=False)
        bilayer = ase_atom * (1, 1, 2)
        x, y, z =  ase_atom.cell.lengths().tolist()
        return bilayer, x, y, z, len(ase_atom)
    elif len(fragments) == 2 and all(len(sublist) == len(fragments[0]) for sublist in fragments):
        cell = ase_atom.get_cell().copy()
        x, y, z =  ase_atom.cell.lengths().tolist()
        bilayer = ase_atom
        return ase_atom, x, y, z, len(fragments[0])
    elif len(fragments) == 2 and not all(len(fragments[0]) for sublist in fragments):
        print("Multi components with different length detected. Will treat as monolayer or stop calculation ")
        cell = ase_atom.get_cell().copy()
        cell[2, 2] = interlayer_dist
        ase_atom.set_cell(cell, scale_atoms=False)
        bilayer = ase_atom * (1, 1, 2)
        x, y, z =  ase_atom.cell.lengths().tolist()
        return bilayer, x, y, z, len(ase_atom)
    else:
        raise("Multi components detected. Check structure")


def find_latest_gulp(base_dir: Optional[str] = None) -> Optional[str]:
    """
    Search for the latest GULP executable in the given base directory.
    Example expected paths: $HOME/src/gulp-6.0/Src/gulp
    """
    base_dir = base_dir or os.path.expandvars("$HOME/src")
    pattern = os.path.join(base_dir, "gulp-*", "Src", "gulp")
    candidates = glob.glob(pattern)

    if not candidates:
        return None

    # Extract version numbers safely (e.g., "gulp-6.2" -> (6,2))
    def version_key(path):
        match = re.search(r"gulp-(\d+)(?:\.(\d+))?", path)
        if match:
            major = int(match.group(1))
            minor = int(match.group(2) or 0)
            return (major, minor)
        return (0, 0)

    # Sort candidates by version
    candidates.sort(key=version_key, reverse=True)
    return candidates[0]  # latest

def optimize_with_gulp(
    ase_atom: Atoms,
    gulp_exe: Optional[str] = None,
    workdir: Optional[str] = None,
    keep_workdir: bool = False,
    timeout: int = 3000000,
    add_c_if_2d: bool = True,
    gin_name: str = "job.gin",
    expected_cif: str = "job.cif",
) -> Atoms:
    """
    Optimize a structure with GULP and return the optimized ASE Atoms.
    """

    # Resolve GULP executable
    gulp_exe = (
        gulp_exe
        or os.environ.get("GULP_EXE")
        or find_latest_gulp()  # ← find the latest version automatically
        or "gulp"  # fallback to PATH
    )

    if not shutil.which(gulp_exe) and not os.path.isfile(gulp_exe):
        raise RuntimeError(f"GULP executable not found: {gulp_exe}")

    # Working directory setup
    owned_tmp = False
    if workdir is None:
        workdir = tempfile.mkdtemp(prefix="gulp_run_")
        owned_tmp = True
    os.makedirs(workdir, exist_ok=True)

    try:
        gin_path = os.path.join(workdir, gin_name)
        gout_path = os.path.join(workdir, "job.gout")
        got_path = os.path.join(workdir, "job.got")

        bonds, mmtypes = mmanalysis.analyze_mm(ase_atom)

        try:
            mmanalysis.write_gin(
                gin_path, ase_atom, bonds, mmtypes,
                add_c_if_2d=add_c_if_2d,
                output_cif=os.path.join(workdir, expected_cif)
            )
        except TypeError:
            mmanalysis.write_gin(gin_path, ase_atom, bonds, mmtypes, add_c_if_2d=add_c_if_2d)


        with open(gin_path, "r") as fin, open(gout_path, "w") as fout:
            proc = subprocess.run(
                [gulp_exe],
                stdin=fin,
                stdout=fout,
                stderr=subprocess.STDOUT,
                cwd=workdir,
                check=False,
                timeout=timeout,
            )

        if proc.returncode != 0:
            tail = ""
            try:
                with open(gout_path, "r") as f:
                    tail = "".join(f.readlines()[-200:])
            except Exception:
                pass
            raise RuntimeError(f"GULP failed (exit code {proc.returncode}).\n{tail}")

        candidate = os.path.join(workdir, expected_cif)
        if not os.path.isfile(candidate):
            cifs = glob.glob(os.path.join(workdir, "*.cif"))
            if cifs:
                candidate = max(cifs, key=os.path.getmtime)
            else:
                raise RuntimeError("No CIF file produced by GULP.")

        opt_atoms = read(candidate)

        if owned_tmp and not keep_workdir:
            shutil.rmtree(workdir, ignore_errors=True)

        return opt_atoms

    finally:
        if owned_tmp and not keep_workdir:
            shutil.rmtree(workdir, ignore_errors=True)


def optimize_with_gulp2(
    ase_atom: Atoms,
    gulp_exe = os.path.expandvars("$HOME/src/gulp-6.0/Src/gulp"), #Optional[str] = None,
    workdir: Optional[str] = None,
    keep_workdir: bool = False,
    timeout: int = 200000,
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

def find_top_peaks(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    min_rel_height: float = 0.05,
    min_separation_deg: float = 0.30,
    max_peaks: Optional[int] = None,
    ) -> List[float]:
    """
    Return peak positions (2θ) using a simple local-maximum + greedy
    non-overlap filter. Intensities are relative, so min_rel_height is fraction of max.
    """
    t = np.asarray(two_theta, float)
    I = np.asarray(intensity, float)
    if t.size < 3:
        return []

    # Normalize
    Imax = float(I.max()) if I.max() > 0 else 1.0
    if Imax == 0:
        return []

    # Candidate local maxima above threshold
    thr = min_rel_height * Imax
    cand_idx = []
    for i in range(1, len(I) - 1):
        if I[i] >= thr and I[i] >= I[i-1] and I[i] >= I[i+1]:
            cand_idx.append(i)

    if not cand_idx:
        return []

    # Sort candidates by intensity (desc), then greedily keep those separated
    cand_idx.sort(key=lambda i: I[i], reverse=True)
    selected: List[int] = []
    for i in cand_idx:
        if all(abs(t[i] - t[j]) >= min_separation_deg for j in selected):
            selected.append(i)
        if max_peaks is not None and len(selected) >= max_peaks:
            break

    # Return in ascending 2θ
    selected.sort(key=lambda i: t[i])
    return [float(t[i]) for i in selected]



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


def plot_overlay_pxrd(exp_tth, exp_I, sim_tth, sim_I, out_png: str):
    offset = 3.0
    plt.figure(figsize=(9, 6))
    plt.plot(exp_tth, exp_I + offset, ls='-', color='red', lw=1.5, label="Experimental")
    plt.plot(sim_tth, sim_I, color='blue', ls='--', lw=1.5, label="Simulated")
    plt.xlabel(r"2$\theta$ (°)", fontsize=12)
    plt.ylabel("Normalized intensity (a.u.)", fontsize=12)
    plt.legend(loc='upper right', frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# ---------------------- Main class ----------------------

class CreateStack:
    """
    Build AA/AB + systematic slips, simulate PXRD, enforce 0.2° grid
    and evaluate both first-peak match and full-pattern residuals.
    """

    def __init__(self, filename: str, interlayer_dist: float = 4.5, output_dir: str = "."):
        self.filename = filename
        self.interlayer_dist = interlayer_dist
        self.output_dir = output_dir
        ensure_dir(output_dir)

        self.base_name = os.path.basename(filename).split(".")[0]
        self.monolayer = read(filename).copy()
        # cell = self.monolayer.get_cell().copy()
        # cell[2, 2] = interlayer_dist
        # self.monolayer.set_cell(cell, scale_atoms=False)
        bilayer, x, y, z, length_monolayer = create_layer( self.monolayer, interlayer_dist)
        # self.bilayer = self.monolayer * (1, 1, 2)
        self.bilayer = bilayer
        self.length_monolayer = length_monolayer

        # x, y, z = self.monolayer.cell.lengths().tolist()
        self.x, self.y, self.z = x, y, z

    # ----- geometry builders -----

    def _make_shifted(self, dx: float, dy: float, dz: float = 0.0) -> Atoms:
        atoms = self.bilayer.copy()
        n = self.length_monolayer
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
            lines.append(f" End of Report for {self.base_name}_{tag}\n")

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
        # NEW: multi-peak matching controls
        n_peaks_to_match: int = 10,
        min_peaks_required: int = 3,
        peak_tolerance_deg: float = 0.10,
        peak_min_rel_height: float = 0.05,
        peak_min_separation_deg: float = 0.30,
    ) -> Dict:
        """
        Simulate PXRD, align to exp (0.2° grid), scan zero-shift & scale analytically,
        compute residuals, and test multi-peak matching: require that at least
        `min_peaks_required` of the first `n_peaks_to_match` experimental peaks are found
        within `peak_tolerance_deg`.
        """
        atoms = optimize_with_gulp(atoms_in) if optimize else atoms_in.copy()
        cif_used = os.path.join(self.output_dir, f"{self.base_name}_{tag}.cif")

        # --- simulate & align ---
        sim_tth_raw, sim_I_raw = xrd_pattern(
            atoms,
            two_theta_min=sim_cfg.get("tth_min", np.min(exp_tth)),
            two_theta_max=sim_cfg.get("tth_max", np.max(exp_tth)),
            wavelength=sim_cfg.get("wavelength", 1.5406),
            fwhm=sim_cfg.get("fwhm", 0.1),
            points=sim_cfg.get("points", 5000),
        )
        sim_on_exp = np.interp(exp_tth, sim_tth_raw, sim_I_raw, left=0.0, right=0.0)

        shifts = np.arange(-zero_shift_range, zero_shift_range + 1e-12, zero_shift_step)
        best = {"Rwp": np.inf, "shift": 0.0, "scale": 1.0, "ycalc": sim_on_exp.copy()}
        for s in shifts:
            sim_shifted = np.interp(exp_tth, sim_tth_raw + s, sim_I_raw, left=0.0, right=0.0)
            scale = scale_least_squares(exp_I, sim_shifted)
            ycalc = np.clip(scale * sim_shifted, 0.0, None)
            r = compute_r_factors(exp_I, ycalc, weights=None, P=0)
            if r["Rwp"] < best["Rwp"]:
                best.update(shift=float(s), scale=float(scale), ycalc=ycalc, **r)

        ycalc = best["ycalc"]
        r_full = compute_r_factors(exp_I, ycalc, weights=None, P=0)
        cos_sim = float(np.dot(exp_I, ycalc) / (np.linalg.norm(exp_I) * np.linalg.norm(ycalc) + 1e-12))
        pear_r = pearson_r(exp_I, ycalc)

        # --- first-peak stats (keep your old metric) ---
        exp_first = first_peak(exp_tth, exp_I)
        sim_first = first_peak(exp_tth, ycalc)
        d_first = abs(exp_first - sim_first) if (exp_first is not None and sim_first is not None) else None

        Rp_peak = None
        if exp_first is not None:
            mask = peak_window_mask(exp_tth, exp_first, half_width=first_peak_window)
            if np.any(mask):
                Rp_peak = compute_r_factors(exp_I[mask], ycalc[mask])["Rp"]

        # --- NEW: multi-peak match on the aligned/processed profiles ---
        exp_peaks = find_top_peaks(
            exp_tth, exp_I,
            min_rel_height=peak_min_rel_height,
            min_separation_deg=peak_min_separation_deg,
            max_peaks=n_peaks_to_match,
        )
        sim_peaks = find_top_peaks(
            exp_tth, ycalc,
            min_rel_height=peak_min_rel_height,
            min_separation_deg=peak_min_separation_deg,
            max_peaks=max(2 * n_peaks_to_match, n_peaks_to_match),  # be generous
        )

        per_peak_deltas: List[Optional[float]] = []
        matched = 0
        used_sim = set()
        for t_exp in exp_peaks:
            if not sim_peaks:
                per_peak_deltas.append(None)
                continue
            # nearest simulated peak not yet used
            diffs = [abs(t_exp - t_sim) if j not in used_sim else np.inf
                    for j, t_sim in enumerate(sim_peaks)]
            jbest = int(np.argmin(diffs))
            d = diffs[jbest]
            if np.isfinite(d):
                per_peak_deltas.append(float(d))
                if d <= peak_tolerance_deg:
                    matched += 1
                    used_sim.add(jbest)
            else:
                per_peak_deltas.append(None)

        # Keep your original criterion, but ALSO require multi-peak consensus.
        first_peak_ok = (d_first is not None) and (d_first <= tol_first_peak_deg)
        multi_peak_ok = (matched >= min_peaks_required)

        is_match = multi_peak_ok  # <- core new behavior
        # If you prefer to keep legacy behavior as fallback, you could do:
        # is_match = multi_peak_ok or first_peak_ok

        # --- report ---
        report = []
        report.append("Report")
        report.append(f"Stacking: {tag}")
        report.append(f"Zero shift applied (deg): {best['shift']:.3f}")
        report.append(f"Scale factor: {best['scale']:.6f}")
        if exp_first is not None:
            report.append(f"Experimental first peak (2θ): {exp_first:.4f} °")
        if sim_first is not None:
            report.append(f"Computed first peak  (2θ): {sim_first:.4f} °")
        if d_first is not None:
            report.append(f"|Δ first peak|: {d_first:.4f} °")
        report.append(f"Matched peaks (out of first {n_peaks_to_match}): {matched}")
        if per_peak_deltas:
            report.append("Per-peak |Δ2θ| for experimental first peaks:")
            report.extend([f"  Peak {i+1}: {d:.4f} °" if d is not None else f"  Peak {i+1}: n/a"
                        for i, d in enumerate(per_peak_deltas)])
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

        txt_path = self._write_report_txt(atoms, tag, report)

        plot_path = None
        if is_match:
            plot_path = os.path.join(self.output_dir, f"{self.base_name}_{tag}_pxrd_best_match.png")
            plot_overlay_pxrd(exp_tth, exp_I, exp_tth, ycalc, plot_path)
            with open(txt_path, "a", encoding="utf-8") as f:
                f.write("\n\nMatch found\n")
            self._write_cif(atoms, f"{tag}_final")
        png_path = os.path.join(self.output_dir, f'{self.base_name}_pxrds_images')
        os.makedirs(png_path, exist_ok=True)
        plot_path = os.path.join(png_path , f"{self.base_name}_{tag}_pxrd.png")
        plot_overlay_pxrd(exp_tth, exp_I, exp_tth, ycalc, plot_path)

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
        matched_peaks=matched,
        per_peak_deltas=per_peak_deltas,
        exp_peaks=exp_peaks,
        sim_peaks=sim_peaks,
    )

    # ---------------------- public API ----------------------

    def search_best_stacking(
        self,
        exp_tth: np.ndarray,
        exp_I: np.ndarray,
        optimize: bool = False,
        tol_first_peak_deg: float = 0.05,
        slip_step: float = 0.5,
        slip_max: float = 8.0,
        sim_cfg: Optional[Dict] = None,
        rwp_threshold: Optional[float] = None,
        # NEW: multi-peak matching defaults (3 of first 5 within 0.10°)
        n_peaks_to_match: int = 5,
        min_peaks_required: int = 3,
        peak_tolerance_deg: float = 0.10,
        peak_min_rel_height: float = 0.05,
        peak_min_separation_deg: float = 0.30,
        ) -> Dict:
        """
            1) Enforce experimental grid to 0.2°
            2) Try AA, AB, then slips (x, y, xy) at 0.5 Å up to 8 Å.
            3) For each: simulate, zero-shift & scale against experimental 0.2° grid,
            compute residuals & first-peak match; stop if |Δ2θ| ≤ tol.
            Optionally also stop if Rwp ≤ rwp_threshold.
            Returns the first structure that meets criteria; otherwise the last tried.
        """
        res_aa = self._simulate_and_match(
            self.create_aa(), "aa", exp_tth, exp_I, optimize, tol_first_peak_deg, sim_cfg,
            n_peaks_to_match=n_peaks_to_match,
            min_peaks_required=min_peaks_required,
            peak_tolerance_deg=peak_tolerance_deg,
            peak_min_rel_height=peak_min_rel_height,
            peak_min_separation_deg=peak_min_separation_deg,
        )
        if res_aa["match"] or (rwp_threshold is not None and res_aa["Rwp"] <= rwp_threshold):
            return res_aa

        res_ab = self._simulate_and_match(
            self.create_ab(), "ab", exp_tth, exp_I, optimize, tol_first_peak_deg, sim_cfg,
            n_peaks_to_match=n_peaks_to_match,
            min_peaks_required=min_peaks_required,
            peak_tolerance_deg=peak_tolerance_deg,
            peak_min_rel_height=peak_min_rel_height,
            peak_min_separation_deg=peak_min_separation_deg,
        )
        if res_ab["match"] or (rwp_threshold is not None and res_ab["Rwp"] <= rwp_threshold):
            return res_ab

        shifts = np.arange(slip_step, slip_max + 1e-9, slip_step)
        last_res = res_ab
        for s in shifts:
            res_x = self._simulate_and_match(
                self._make_shifted(s, 0.0), f"x_{s:.1f}", exp_tth, exp_I, optimize, tol_first_peak_deg, sim_cfg,
                n_peaks_to_match=n_peaks_to_match,
                min_peaks_required=min_peaks_required,
                peak_tolerance_deg=peak_tolerance_deg,
                peak_min_rel_height=peak_min_rel_height,
                peak_min_separation_deg=peak_min_separation_deg,
            )
            last_res = res_x
            if res_x["match"] or (rwp_threshold is not None and res_x["Rwp"] <= rwp_threshold):
                return res_x

            res_y = self._simulate_and_match(
                self._make_shifted(0.0, s), f"y_{s:.1f}", exp_tth, exp_I, optimize, tol_first_peak_deg, sim_cfg,
                n_peaks_to_match=n_peaks_to_match,
                min_peaks_required=min_peaks_required,
                peak_tolerance_deg=peak_tolerance_deg,
                peak_min_rel_height=peak_min_rel_height,
                peak_min_separation_deg=peak_min_separation_deg,
            )
            last_res = res_y
            if res_y["match"] or (rwp_threshold is not None and res_y["Rwp"] <= rwp_threshold):
                return res_y

            res_xy = self._simulate_and_match(
                self._make_shifted(s, s), f"xy_{s:.1f}", exp_tth, exp_I, optimize, tol_first_peak_deg, sim_cfg,
                n_peaks_to_match=n_peaks_to_match,
                min_peaks_required=min_peaks_required,
                peak_tolerance_deg=peak_tolerance_deg,
                peak_min_rel_height=peak_min_rel_height,
                peak_min_separation_deg=peak_min_separation_deg,
            )
            last_res = res_xy
            if res_xy["match"] or (rwp_threshold is not None and res_xy["Rwp"] <= rwp_threshold):
                return res_xy

        return last_res


    def search_best_stacking2(
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


def simulate_and_plot_pxrd_from_exp(
    ase_atom: Atoms,
    exp_two_theta: np.ndarray,
    exp_intensity: np.ndarray,
    out_png: str = "pxrd_overlay.png",
    sim_cfg: Optional[Dict] = None,
    normalize_exp: bool = True,
    zero_shift_range: float = 0.20,
    zero_shift_step: float = 0.01,
    save_csv: Optional[str] = None,
) -> Dict:
    """
    Simulate PXRD for an ASE Atoms object and overlay it with experimental data.

    Steps
    -----
    1) Ensure experimental grid is 0.2° (resamples if needed) and (optionally) normalize to max=1.
    2) Simulate PXRD via pymatgen's XRDCalculator (broadened using xrd_pattern).
    3) Scan a small zero shift; scale simulated intensities by least squares.
    4) Plot overlay and (optionally) save a CSV with aligned arrays.

    Parameters
    ----------
    ase_atom : Atoms
        Structure to simulate.
    exp_two_theta, exp_intensity : array-like
        Experimental 2θ and intensity arrays.
    out_png : str
        Output path for the overlay PNG.
    sim_cfg : dict, optional
        Keys: tth_min, tth_max, wavelength, fwhm, points (passed to xrd_pattern).
    normalize_exp : bool
        If True, normalize experimental intensities to max=1 after resampling.
    zero_shift_range : float
        Scan ± this many degrees for best zero shift.
    zero_shift_step : float
        Step size for zero-shift scan.
    save_csv : str or None
        If provided, write a CSV with columns: two_theta, I_exp, I_sim_best.

    Returns
    -------
    dict
        {
          "two_theta": exp_tth (0.2° grid),
          "I_exp": normalized experimental intensities,
          "I_sim": best-fit simulated intensities on the same grid,
          "zero_shift": best zero shift (deg),
          "scale": scale factor,
          "Rp", "Rwp", "Rexp", "GoF": profile statistics,
          "cosine": cosine similarity,
          "pearson": Pearson r,
          "png": out_png,
          "csv": save_csv or None
        }
    """
    sim_cfg = sim_cfg or {}

    # -- 1) Put experimental data on a 0.2° grid
    exp_tth, exp_I, _ = ensure_exp_step_0p2(np.asarray(exp_two_theta), np.asarray(exp_intensity))
    if normalize_exp and exp_I.max() > 0:
        exp_I = exp_I / exp_I.max()

    # -- 2) Simulate PXRD for the structure
    tth_min = sim_cfg.get("tth_min", float(np.min(exp_tth)))
    tth_max = sim_cfg.get("tth_max", float(np.max(exp_tth)))
    wavelength = sim_cfg.get("wavelength", 1.5406)
    fwhm = sim_cfg.get("fwhm", 0.10)
    points = sim_cfg.get("points", 5000)

    sim_tth_raw, sim_I_raw = xrd_pattern(
        ase_atom,
        two_theta_min=tth_min,
        two_theta_max=tth_max,
        wavelength=wavelength,
        fwhm=fwhm,
        points=points,
    )

    # Interpolate simulated pattern to the experimental (0.2°) grid
    sim_on_exp = np.interp(exp_tth, sim_tth_raw, sim_I_raw, left=0.0, right=0.0)

    # -- 3) Scan zero shift & compute least-squares scale
    shifts = np.arange(-zero_shift_range, zero_shift_range + 1e-12, zero_shift_step)
    best = {"Rwp": np.inf, "shift": 0.0, "scale": 1.0, "ycalc": sim_on_exp.copy()}

    for s in shifts:
        shifted = np.interp(exp_tth, sim_tth_raw + s, sim_I_raw, left=0.0, right=0.0)
        scale = scale_least_squares(exp_I, shifted)
        ycalc = np.clip(scale * shifted, 0.0, None)
        r = compute_r_factors(exp_I, ycalc)
        if r["Rwp"] < best["Rwp"]:
            best.update(shift=float(s), scale=float(scale), ycalc=ycalc, **r)

    ycalc = best["ycalc"]
    r_full = compute_r_factors(exp_I, ycalc)
    cos_sim = float(np.dot(exp_I, ycalc) / (np.linalg.norm(exp_I) * np.linalg.norm(ycalc) + 1e-12))
    pear = pearson_r(exp_I, ycalc)

    # -- 4) Plot overlay and optionally save CSV
    plot_overlay_pxrd(exp_tth, exp_I, exp_tth, ycalc, out_png)

    if save_csv is not None:
        try:
            import csv
            with open(save_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["two_theta_deg", "I_exp_norm", "I_sim_best"])
                for t, ie, ic in zip(exp_tth, exp_I, ycalc):
                    writer.writerow([f"{t:.4f}", f"{ie:.6f}", f"{ic:.6f}"])
        except Exception:
            # If CSV save fails, just continue—PNG and return dict still useful
            save_csv = None

    return dict(
        two_theta=exp_tth,
        I_exp=exp_I,
        I_sim=ycalc,
        zero_shift=best["shift"],
        scale=best["scale"],
        Rp=r_full["Rp"],
        Rwp=r_full["Rwp"],
        Rexp=r_full["Rexp"],
        GoF=r_full["GoF"],
        cosine=cos_sim,
        pearson=pear,
        png=out_png,
        csv=save_csv,
        cfg=dict(wavelength=wavelength, fwhm=fwhm, tth_min=tth_min, tth_max=tth_max, points=points),
    )
