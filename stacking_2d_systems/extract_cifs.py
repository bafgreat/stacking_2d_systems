#!/usr/bin/env python3
import os
import argparse
from ase import Atoms
from stacking_2d_systems import file_typer

def get_ase_atoms(contents, base_name, stack='aa'):
    symbols = []
    positions = []
    cell = []
    aa_section = file_typer.get_section(contents,
                                        f"Calculation Report for: {base_name}_{stack}",
                                        "PXRD and Similarity Report",
                                        6,
                                        -1)

    aa_atom = file_typer.get_section(aa_section, "Atom", "End", 2, -1)
    for atoms in aa_atom:
        data = atoms.split()
        symbols.append(data[0])
        positions.append([float(i) for i in data[1:]])
    lattice_data = file_typer.get_section(aa_section, "Lattice", "End", 1, -1)
    for lat in lattice_data:
        lattice = lat.split()
        cell.append([float(i) for i in lattice[1:]])
    ase_atoms = Atoms(symbols=symbols,
                      positions=positions,
                      cell=cell,
                      pbc=True
                      )
    return ase_atoms

def get_data(contents, base_name):
    data_dic = {}
    for line in contents:
        if "Stacking :" in line:
            stack = line.split()[-1]
            tmp = {}
            section = file_typer.get_section(contents,
                                                f"Calculation Report for: {base_name}_{stack}",
                                                " Intensity 2theta",
                                                6,
                                                -1)
            for data in section:
                if "Cosine similarity:" in data:
                    cosine = float(data.split()[-1])
                    tmp['cosine'] = cosine
                elif "Pearson r:" in data:
                    pearson = float(data.split()[-1])
                    tmp['pearson'] = pearson

            data_dic[stack] = tmp
    return data_dic


# def compile_data(filename, stacked=None):
#     contents = file_typer.get_contents(filename)
#     base_name = os.path.basename(filename).split('.')[0]
#     data_dic = get_data(contents, base_name)
#     filtered = {k: v for k, v in data_dic.items() if k not in ('aa', 'ab')}
#     best_key = max(filtered, key=lambda k: filtered[k]['cosine'])

#     aa_atom = get_ase_atoms(contents, base_name, stack='aa')
#     ab_atom = get_ase_atoms(contents, base_name, stack='ab')
#     slip_atom = get_ase_atoms(contents, base_name, stack=best_key)
#     aa_atom.write(f"{base_name}_aa.cif")
#     ab_atom.write(f"{base_name}_ab.cif")
#     slip_atom.write(f"{base_name}_best_slip_{best_key}.cif")
#     if stacked is not None:
#         custom_atom = get_ase_atoms(contents, base_name, stack=stacked)
#         custom_atom.write(f"{base_name}_slip_{stacked}.cif")

#     return data_dic

def compile_data(filename, stacked=None, stacked_only=False):
    contents = file_typer.get_contents(filename)
    base_name = os.path.basename(filename).split('.')[0]

    data_dic = get_data(contents, base_name)

    # If user only wants specific stackings
    if stacked_only and stacked:
        stacks = [s.strip() for s in stacked.split(",")]
        for st in stacks:
            atom = get_ase_atoms(contents, base_name, stack=st)
            atom.write(f"{base_name}_slip_{st}.cif")
        return data_dic

    # Normal workflow: AA, AB, best slip
    filtered = {k: v for k, v in data_dic.items() if k not in ("aa", "ab")}
    best_key = max(filtered, key=lambda k: filtered[k]["cosine"])

    # write AA, AB, and best slip
    aa_atom = get_ase_atoms(contents, base_name, "aa")
    ab_atom = get_ase_atoms(contents, base_name, "ab")
    slip_atom = get_ase_atoms(contents, base_name, best_key)

    aa_atom.write(f"{base_name}_aa.cif")
    ab_atom.write(f"{base_name}_ab.cif")
    slip_atom.write(f"{base_name}_best_slip_{best_key}.cif")

    # If user added custom stackings, also output them
    if stacked:
        stacks = [s.strip() for s in stacked.split(",")]
        for st in stacks:
            atom = get_ase_atoms(contents, base_name, stack=st)
            atom.write(f"{base_name}_slip_{st}.cif")

    return data_dic


def main():
    parser = argparse.ArgumentParser(
        description="Extract AA, AB, best slip, and optionally selected slip configurations."
    )

    parser.add_argument(
        "-f", "--file",
        required=True,
        help="Path to PXRD report file (e.g., BPDA.txt)"
    )

    parser.add_argument(
        "-s", "--stacked",
        help="Comma-separated list of slip systems to extract (e.g., x_2.0,y_3.5)"
    )

    parser.add_argument(
        "--stacked-only",
        action="store_true",
        help="Only write the stackings provided in --stacked; skip AA, AB, and best."
    )

    args = parser.parse_args()

    compile_data(args.file, stacked=args.stacked, stacked_only=args.stacked_only)


# filename = "../tests/BPDA.txt"

# compile_data(filename)


# get_ase_atoms("../tests/BPDA.txt")