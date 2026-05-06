"""
extract_mol2_features.py
------------------------
Lee todos los archivos .mol2 de una carpeta y genera un CSV con features por átomo.

Uso:
    python extract_mol2_features.py --input carpeta_mol2/ --output features.csv

Dependencias:
    pip install rdkit pandas numpy
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from rdkit import Chem
except ImportError:
    sys.exit("RDKit no encontrado. Instalalo con: pip install rdkit")

# Masas atómicas estándar (g/mol)
ATOMIC_MASSES = {
    "H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999,
    "F": 18.998, "P": 30.974, "S": 32.06,  "Cl": 35.45,
    "Br": 79.904, "I": 126.904, "B": 10.811, "Si": 28.085,
    "Se": 78.971, "Fe": 55.845, "Zn": 65.38, "Ca": 40.078,
    "Mg": 24.305, "Na": 22.990, "K": 39.098,
}

# Tipos Tripos considerados planos
PLANAR_TYPES = {"C.2", "C.ar", "N.ar", "N.2", "N.pl3", "O.2", "S.2", "N.am"}


def parse_mol2_atoms(mol2_path: Path) -> dict:
    atoms = {}
    in_atom_block = False

    with open(mol2_path, "r") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("@<TRIPOS>ATOM"):
                in_atom_block = True
                continue
            if line.startswith("@<TRIPOS>") and in_atom_block:
                break
            if in_atom_block and line.strip():
                parts = line.split()
                if len(parts) >= 6:
                    atom_id = int(parts[0])
                    x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                    tripos_type = parts[5] if len(parts) > 5 else "?"
                    atoms[atom_id] = {"tripos_type": tripos_type, "xyz": (x, y, z)}
    return atoms


def angle_deg(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-12)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))


def bond_length(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def is_planar_by_geometry(center_xyz, neighbor_xyzs, threshold=0.1):
    if len(neighbor_xyzs) < 3:
        return True
    pts = np.array([center_xyz] + list(neighbor_xyzs))
    pts -= pts.mean(axis=0)
    _, s, _ = np.linalg.svd(pts)
    return float(s[-1]) < threshold


def extract_features(mol2_path: Path) -> list:
    mol_name = mol2_path.stem
    tripos_data = parse_mol2_atoms(mol2_path)

    mol = Chem.MolFromMol2File(str(mol2_path), removeHs=False)
    if mol is None:
        print(f"  [WARN] RDKit no pudo leer: {mol2_path.name} — saltando.")
        return []

    conf = mol.GetConformer()
    num_atoms = mol.GetNumAtoms()

    def get_tripos(rdkit_idx):
        return tripos_data.get(rdkit_idx + 1, {})

    # n_dihedrals por átomo
    dihedral_counts = {i: 0 for i in range(num_atoms)}
    for bond_bc in mol.GetBonds():
        b = bond_bc.GetBeginAtomIdx()
        c = bond_bc.GetEndAtomIdx()
        for a in [n.GetIdx() for n in mol.GetAtomWithIdx(b).GetNeighbors() if n.GetIdx() != c]:
            for d in [n.GetIdx() for n in mol.GetAtomWithIdx(c).GetNeighbors() if n.GetIdx() != b]:
                for idx in (a, b, c, d):
                    dihedral_counts[idx] += 1

    rows = []
    for i in range(num_atoms):
        atom = mol.GetAtomWithIdx(i)
        tdata = get_tripos(i)
        tripos_type = tdata.get("tripos_type", "?")
        xyz_i = tdata.get("xyz", conf.GetAtomPosition(i))

        element = tripos_type.split(".")[0] if "." in tripos_type else atom.GetSymbol()
        mass = ATOMIC_MASSES.get(element, 0.0)

        neighbors = atom.GetNeighbors()
        coordination = len(neighbors)

        neighbor_info = []
        bond_lengths = []
        neighbor_xyzs = []
        elem_counts = {"C": 0, "H": 0, "O": 0, "N": 0}

        for nb in neighbors:
            nb_idx = nb.GetIdx()
            nb_tdata = get_tripos(nb_idx)
            nb_type = nb_tdata.get("tripos_type", "?")
            nb_elem = nb_type.split(".")[0] if "." in nb_type else nb.GetSymbol()
            bond = mol.GetBondBetweenAtoms(i, nb_idx)
            bond_type = bond.GetBondTypeAsDouble()
            neighbor_info.append(f"{nb_elem}{int(bond_type) if bond_type == int(bond_type) else bond_type}")

            if nb_elem in elem_counts:
                elem_counts[nb_elem] += 1

            nb_xyz = nb_tdata.get("xyz", tuple(conf.GetAtomPosition(nb_idx)))
            neighbor_xyzs.append(nb_xyz)
            bond_lengths.append(bond_length(xyz_i, nb_xyz))

        neighbor_config = "-".join(sorted(neighbor_info))
        avg_bond_len = float(np.mean(bond_lengths)) if bond_lengths else 0.0

        angles = []
        nb_list = list(neighbors)
        for j in range(len(nb_list)):
            for k in range(j + 1, len(nb_list)):
                a_xyz = get_tripos(nb_list[j].GetIdx()).get("xyz", tuple(conf.GetAtomPosition(nb_list[j].GetIdx())))
                c_xyz = get_tripos(nb_list[k].GetIdx()).get("xyz", tuple(conf.GetAtomPosition(nb_list[k].GetIdx())))
                angles.append(angle_deg(a_xyz, xyz_i, c_xyz))
        avg_angle = float(np.mean(angles)) if angles else 0.0

        if tripos_type in PLANAR_TYPES:
            planar = True
        elif coordination >= 3:
            planar = is_planar_by_geometry(xyz_i, neighbor_xyzs)
        else:
            planar = False

        rows.append({
            "molecule": mol_name,
            "atom_id": i + 1,
            "atom_name": tripos_data.get(i + 1, {}).get("tripos_type", "?").split(".")[0] + str(i + 1),
            "tripos_type": tripos_type,
            "element": element,
            "mass": round(mass, 4),
            "coordination": coordination,
            "neighbor_config": neighbor_config,
            "n_dihedrals": dihedral_counts[i],
            "is_planar": planar,
            "avg_bond_len_A": round(avg_bond_len, 4),
            "avg_angle_deg": round(avg_angle, 4),
            "C_count": elem_counts["C"],
            "H_count": elem_counts["H"],
            "O_count": elem_counts["O"],
            "N_count": elem_counts["N"],
        })

    return rows


def main():
    parser = argparse.ArgumentParser(description="Extrae features atómicos de archivos .mol2")
    parser.add_argument("--input",  "-i", required=True, help="Carpeta con archivos .mol2")
    parser.add_argument("--output", "-o", default="features.csv", help="Archivo CSV de salida")
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        sys.exit(f"Error: '{input_dir}' no es una carpeta válida.")

    mol2_files = sorted(input_dir.glob("*.mol2"))
    if not mol2_files:
        sys.exit(f"No se encontraron archivos .mol2 en '{input_dir}'.")

    print(f"Encontrados {len(mol2_files)} archivos .mol2 en '{input_dir}'")

    all_rows = []
    for mol2_file in mol2_files:
        print(f"  Procesando: {mol2_file.name} ...", end=" ")
        rows = extract_features(mol2_file)
        all_rows.extend(rows)
        print(f"{len(rows)} átomos")

    if not all_rows:
        sys.exit("No se pudo extraer ningún átomo. Verificá los archivos .mol2.")

    df = pd.DataFrame(all_rows, columns=[
        "molecule", "atom_id", "atom_name", "tripos_type", "element", "mass",
        "coordination", "neighbor_config", "n_dihedrals", "is_planar",
        "avg_bond_len_A", "avg_angle_deg",
        "C_count", "H_count", "O_count", "N_count",
    ])

    output_path = Path(args.output)
    df.to_csv(output_path, index=False)
    print(f"\n✓ CSV guardado en: {output_path}")
    print(f"  {len(df)} filas · {df['molecule'].nunique()} moléculas")


if __name__ == "__main__":
    main()
