#!/usr/bin/env python3
"""
Usage:
    python make_primitive.py input.cif output_primitive.cif

Requirements:
    pip install ase spglib
"""

import argparse
import numpy as np
from ase.io import read, write
from ase import Atoms
import spglib


def get_primitive_from_ase(atoms: Atoms,
                           symprec: float = 1e-3,
                           angle_tolerance: float = 5.0,
                           no_idealize: bool = False) -> Atoms:
    """
    Convert an ASE Atoms object to its primitive cell using spglib.

    Parameters
    ----------
    atoms : Atoms
        Input structure.
    symprec : float
        Symmetry tolerance in Å.
    angle_tolerance : float
        Symmetry tolerance in degrees.
    no_idealize : bool
        If True, do not idealize the structure (keep original metric as much as possible).

    Returns
    -------
    prim : Atoms
        Primitive cell as an ASE Atoms object.
    """
    # Build spglib cell: (lattice, fractional_positions, atomic_numbers)
    lattice = atoms.cell.array
    scaled_positions = atoms.get_scaled_positions()
    numbers = atoms.get_atomic_numbers()

    spglib_cell = (lattice, scaled_positions, numbers)

    prim_cell = spglib.standardize_cell(
        spglib_cell,
        to_primitive=True,
        no_idealize=no_idealize,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
    )

    if prim_cell is None:
        raise RuntimeError("spglib could not find a primitive cell; "
                           "try adjusting symprec/angle_tolerance.")

    lattice_p, scaled_positions_p, numbers_p = prim_cell

    prim_atoms = Atoms(
        numbers=numbers_p,
        cell=lattice_p,
        pbc=True,
    )
    prim_atoms.set_scaled_positions(scaled_positions_p)

    # Carry over some metadata if present
    prim_atoms.info.update(atoms.info)
    return prim_atoms


def main():
    parser = argparse.ArgumentParser(
        description="Read a CIF and reduce it to the smallest repeating unit (primitive cell)."
    )
    parser.add_argument("input_cif", help="Input CIF file")
    parser.add_argument("output_cif", help="Output CIF file for primitive cell")
    parser.add_argument("--symprec", type=float, default=1e-3,
                        help="Symmetry tolerance in Å (default: 1e-3)")
    parser.add_argument("--angle-tol", type=float, default=5.0,
                        help="Angle tolerance in degrees (default: 5.0)")
    parser.add_argument("--no-idealize", action="store_true",
                        help="Do not idealize the cell (keep original metric)")

    args = parser.parse_args()

    # Read CIF
    atoms = read(args.input_cif)

    # Get primitive cell
    prim_atoms = get_primitive_from_ase(
        atoms,
        symprec=args.symprec,
        angle_tolerance=args.angle_tol,
        no_idealize=args.no_idealize,
    )

    # Write primitive cell to CIF
    write(args.output_cif, prim_atoms)
    print(f"Primitive cell written to: {args.output_cif}")
    print(f"Original cell has {len(atoms)} atoms, primitive has {len(prim_atoms)} atoms.")


if __name__ == "__main__":
    main()
