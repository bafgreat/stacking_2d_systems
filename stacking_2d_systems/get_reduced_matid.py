import argparse
import ase.io
from ase.visualize import view
from matid.symmetry import SymmetryAnalyzer


def matid_sbu(system):
    analyzer = SymmetryAnalyzer(system)
    conventional_cell = analyzer.get_conventional_system()
    print(analyzer.get_material_id())
    return conventional_cell


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
    atoms = ase.io.read(args.input_cif)
    conventional_cell = matid_sbu(atoms)

    conventional_cell.write(args.output_cif)
    print(f"Primitive cell written to: {args.output_cif}")
    print(f"Original cell has {len(atoms)} atoms, primitive has {len(conventional_cell )} atoms.")


if __name__ == "__main__":
    main()


