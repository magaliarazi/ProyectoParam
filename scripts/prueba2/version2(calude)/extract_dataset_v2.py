"""
extract_dataset_v2.py
---------------------
Lee archivos .mol2 e .itp (mismo nombre base) de una carpeta y genera un
único CSV con features por átomo + atomtype y charge del .itp como targets.

Cambios respecto a v1:
    - Agrega columna 'bonded_to_element': elemento del vecino directo
      (para H con coordination=1, es el elemento al que está unido)
      Necesaria para el split AA/UA correcto.

Uso:
    python extract_dataset_v2.py --input carpeta/ --output dataset.csv

Dependencias:
    pip install pandas numpy
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

ATOMIC_MASSES = {
    "H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999,
    "F": 18.998, "P": 30.974, "S": 32.06,  "Cl": 35.45,
    "Br": 79.904, "I": 126.904,
}

PLANAR_TYPES = {"C.2", "C.ar", "N.ar", "N.2", "N.pl3", "O.2", "S.2", "N.am"}


# =============================================================================
# PARSER MOL2
# =============================================================================

def parse_mol2(mol2_path):
    """Devuelve (atoms dict, bonds list)."""
    atoms = {}
    bonds = []
    in_atom = in_bond = False

    with open(mol2_path) as f:
        for line in f:
            line = line.strip()

            if line.startswith("@<TRIPOS>ATOM"):
                in_atom, in_bond = True, False
                continue
            if line.startswith("@<TRIPOS>BOND"):
                in_atom, in_bond = False, True
                continue
            if line.startswith("@<TRIPOS>"):
                in_atom = in_bond = False
                continue

            if in_atom and line:
                parts = line.split()
                idx          = int(parts[0])
                atom_name    = parts[1]
                x, y, z      = map(float, parts[2:5])
                tripos_type  = parts[5]
                element      = ''.join(c for c in tripos_type.split('.')[0] if c.isalpha())

                atoms[idx] = {
                    "name":    atom_name,
                    "type":    tripos_type,
                    "element": element,
                    "xyz":     np.array([x, y, z]),
                }

            if in_bond and line:
                parts = line.split()
                bonds.append((int(parts[1]), int(parts[2]), parts[3]))

    return atoms, bonds


# =============================================================================
# PARSER ITP
# =============================================================================

def parse_itp(itp_path):
    """Devuelve dict {atom_idx: {atomtype, charge}}."""
    atoms    = {}
    in_atoms = False

    with open(itp_path) as f:
        for line in f:
            line = line.strip()

            if line.startswith("[ atoms ]"):
                in_atoms = True
                continue
            if line.startswith("["):
                in_atoms = False
                continue
            if not line or line.startswith(";"):
                continue

            if in_atoms:
                parts = line.split()
                atoms[int(parts[0])] = {
                    "atomtype": parts[1],
                    "charge":   float(parts[6]),
                }

    return atoms


# =============================================================================
# GEOMETRÍA
# =============================================================================

def bond_length(a, b):
    return float(np.linalg.norm(a - b))


def angle_deg(a, b, c):
    """Ángulo a-b-c en grados (b es el vértice)."""
    ba = a - b
    bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-12)
    return math.degrees(math.acos(np.clip(cos, -1.0, 1.0)))


def is_planar(center, neighbors):
    if len(neighbors) < 3:
        return False
    pts = np.array([center] + neighbors)
    pts -= pts.mean(axis=0)
    _, s, _ = np.linalg.svd(pts)
    return bool(s[-1] < 0.1)


# =============================================================================
# GRAFO
# =============================================================================

def build_graph(atoms, bonds):
    graph = {i: [] for i in atoms}
    for a1, a2, btype in bonds:
        graph[a1].append((a2, btype))
        graph[a2].append((a1, btype))
    return graph


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_features(mol2_path, itp_path):
    mol_name        = mol2_path.stem
    atoms, bonds    = parse_mol2(mol2_path)
    itp             = parse_itp(itp_path)
    graph           = build_graph(atoms, bonds)

    rows = []

    for i, atom in atoms.items():
        neighbors    = graph[i]
        xyz          = atom["xyz"]
        element      = atom["element"]

        # ── vecinos ──────────────────────────────────────────────────────────
        neighbor_xyz = []
        bond_lengths = []
        elem_counts  = {"C": 0, "H": 0, "O": 0, "N": 0}

        for nb_idx, _ in neighbors:
            nb = atoms[nb_idx]
            neighbor_xyz.append(nb["xyz"])
            bond_lengths.append(bond_length(xyz, nb["xyz"]))
            if nb["element"] in elem_counts:
                elem_counts[nb["element"]] += 1

        # ── ángulos ───────────────────────────────────────────────────────────
        angles = [
            angle_deg(atoms[neighbors[j][0]]["xyz"],
                      xyz,
                      atoms[neighbors[k][0]]["xyz"])
            for j in range(len(neighbors))
            for k in range(j + 1, len(neighbors))
        ]

        avg_angle = float(np.mean(angles)) if angles else 0.0
        avg_bond  = float(np.mean(bond_lengths)) if bond_lengths else 0.0
        planar    = atom["type"] in PLANAR_TYPES or is_planar(xyz, neighbor_xyz)

        # ── bonded_to_element ─────────────────────────────────────────────────
        # Para átomos con un solo vecino (ej: H, halógenos terminales):
        #   → el elemento de ese único vecino.
        # Para átomos con varios vecinos:
        #   → lista separada por comas (info de contexto, no usada en split).
        if len(neighbors) == 1:
            bonded_to_element = atoms[neighbors[0][0]]["element"]
        elif len(neighbors) > 1:
            bonded_to_element = ",".join(
                atoms[nb_idx]["element"] for nb_idx, _ in neighbors
            )
        else:
            bonded_to_element = ""

        # ── targets del ITP ───────────────────────────────────────────────────
        itp_atom = itp.get(i, {})

        rows.append({
            "molecule":          mol_name,
            "atom_id":           i,
            "atom_name":         atom["name"],
            "tripos_type":       atom["type"],
            "element":           element,
            "mass":              ATOMIC_MASSES.get(element, 0.0),
            "coordination":      len(neighbors),
            "avg_bond_len_A":    round(avg_bond, 4),
            "avg_angle_deg":     round(avg_angle, 4),
            "is_planar":         planar,
            "C_count":           elem_counts["C"],
            "H_count":           elem_counts["H"],
            "O_count":           elem_counts["O"],
            "N_count":           elem_counts["N"],
            "bonded_to_element": bonded_to_element,   # ← nuevo
            "atomtype":          itp_atom.get("atomtype"),
            "charge":            itp_atom.get("charge"),
        })

    return rows


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extrae features por átomo de archivos .mol2/.itp"
    )
    parser.add_argument("-i", "--input",  required=True, help="Carpeta con .mol2 y .itp")
    parser.add_argument("-o", "--output", default="dataset.csv", help="CSV de salida")
    args = parser.parse_args()

    input_dir = Path(args.input)
    all_rows  = []

    mol2_files = sorted(input_dir.glob("*.mol2"))
    if not mol2_files:
        print("⚠️  No se encontraron archivos .mol2 en", input_dir)
        return

    for mol2 in mol2_files:
        itp = mol2.with_suffix(".itp")
        if not itp.exists():
            print(f"  ⚠️  Sin .itp para {mol2.name}, omitiendo.")
            continue

        print(f"  ✔ Procesando: {mol2.name}")
        all_rows.extend(extract_features(mol2, itp))

    if not all_rows:
        print("⚠️  No se generaron filas. Revisá los archivos de entrada.")
        return

    df = pd.DataFrame(all_rows).dropna()
    df.to_csv(args.output, index=False)

    print(f"\n✔ Dataset listo: {args.output}")
    print(f"  Filas:    {len(df)}")
    print(f"  Columnas: {list(df.columns)}")


if __name__ == "__main__":
    main()