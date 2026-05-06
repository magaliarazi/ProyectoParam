"""
extract_dataset.py
------------------
Lee archivos .mol2 e .itp (mismo nombre base) de una carpeta y genera un
único CSV con features por átomo + atomtype y charge del .itp como targets.

Uso:
    python extract_dataset.py --input carpeta/ --output dataset.csv

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
    from rdkit.Chem import rdPartialCharges, rdMolDescriptors
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

# Elementos electronegativos
ELECTRONEGATIVE = {"N", "O", "F", "Cl", "S", "Br", "I"}


def parse_mol2_atoms(mol2_path: Path) -> dict:
    """Parsea @<TRIPOS>ATOM: {atom_id (1-based): {tripos_type, xyz, atom_name}}"""
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
                    atom_id     = int(parts[0])
                    atom_name   = parts[1]
                    x, y, z     = float(parts[2]), float(parts[3]), float(parts[4])
                    tripos_type = parts[5]
                    atoms[atom_id] = {
                        "atom_name":   atom_name,
                        "tripos_type": tripos_type,
                        "xyz":         (x, y, z),
                    }
    return atoms


def parse_itp_atoms(itp_path: Path) -> dict:
    """Parsea [ atoms ] del .itp: {atom_name: {atomtype, charge}}"""
    atoms = {}
    in_atoms_block = False

    with open(itp_path, "r") as f:
        for line in f:
            line = line.strip()
            if "[ atoms ]" in line:
                in_atoms_block = True
                continue
            if in_atoms_block and line.startswith("["):
                break
            if not line or line.startswith(";"):
                continue
            if in_atoms_block:
                parts = line.split()
                if len(parts) >= 7:
                    atom_name = parts[4]
                    atomtype  = parts[1]
                    charge    = float(parts[6])
                    atoms[atom_name] = {"atomtype": atomtype, "charge": charge}

    return atoms


def angle_deg(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-12)
    return math.degrees(math.acos(np.clip(cos_angle, -1.0, 1.0)))


def bond_length(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def is_planar_by_geometry(center_xyz, neighbor_xyzs, threshold=0.1):
    if len(neighbor_xyzs) < 3:
        return True
    pts = np.array([center_xyz] + list(neighbor_xyzs))
    pts -= pts.mean(axis=0)
    _, s, _ = np.linalg.svd(pts)
    return float(s[-1]) < threshold


def extract_molecule(mol2_path: Path, itp_path: Path) -> list:
    mol_name    = mol2_path.stem
    tripos_data = parse_mol2_atoms(mol2_path)
    itp_data    = parse_itp_atoms(itp_path) if itp_path and itp_path.exists() else {}

    mol = Chem.MolFromMol2File(str(mol2_path), removeHs=False)
    if mol is None:
        print(f"  [WARN] RDKit no pudo leer: {mol2_path.name} — saltando.")
        return []

    # Calcular cargas de Gasteiger
    try:
        rdPartialCharges.ComputeGasteigerCharges(mol)
        has_gasteiger = True
    except Exception:
        has_gasteiger = False

    conf      = mol.GetConformer()
    num_atoms = mol.GetNumAtoms()
    ring_info = mol.GetRingInfo()

    def get_tripos(rdkit_idx):
        return tripos_data.get(rdkit_idx + 1, {})

    def get_element(rdkit_idx):
        tdata = get_tripos(rdkit_idx)
        tt = tdata.get("tripos_type", "?")
        return tt.split(".")[0] if "." in tt else mol.GetAtomWithIdx(rdkit_idx).GetSymbol()

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
    unmatched = []

    for i in range(num_atoms):
        atom        = mol.GetAtomWithIdx(i)
        tdata       = get_tripos(i)
        tripos_type = tdata.get("tripos_type", "?")
        atom_name   = tdata.get("atom_name", f"?{i+1}")
        xyz_i       = tdata.get("xyz", tuple(conf.GetAtomPosition(i)))

        element = tripos_type.split(".")[0] if "." in tripos_type else atom.GetSymbol()
        mass    = ATOMIC_MASSES.get(element, 0.0)

        # --- Vecinos directos ---
        neighbors    = atom.GetNeighbors()
        coordination = len(neighbors)

        neighbor_info  = []
        bond_lengths   = []
        neighbor_xyzs  = []
        elem_counts    = {"C": 0, "H": 0, "O": 0, "N": 0}
        bond_type_counts = {"single": 0, "double": 0, "aromatic": 0, "other": 0}
        n_electroneg_1  = 0

        for nb in neighbors:
            nb_idx   = nb.GetIdx()
            nb_tdata = get_tripos(nb_idx)
            nb_type  = nb_tdata.get("tripos_type", "?")
            nb_elem  = nb_type.split(".")[0] if "." in nb_type else nb.GetSymbol()
            bond     = mol.GetBondBetweenAtoms(i, nb_idx)
            bt       = bond.GetBondTypeAsDouble()
            neighbor_info.append(f"{nb_elem}{int(bt) if bt == int(bt) else bt}")

            if nb_elem in elem_counts:
                elem_counts[nb_elem] += 1
            if nb_elem in ELECTRONEGATIVE:
                n_electroneg_1 += 1

            # Conteo por tipo de enlace
            btype = bond.GetBondType()
            if btype == Chem.BondType.SINGLE:
                bond_type_counts["single"] += 1
            elif btype == Chem.BondType.DOUBLE:
                bond_type_counts["double"] += 1
            elif btype == Chem.BondType.AROMATIC:
                bond_type_counts["aromatic"] += 1
            else:
                bond_type_counts["other"] += 1

            nb_xyz = nb_tdata.get("xyz", tuple(conf.GetAtomPosition(nb_idx)))
            neighbor_xyzs.append(nb_xyz)
            bond_lengths.append(bond_length(xyz_i, nb_xyz))

        neighbor_config = "-".join(sorted(neighbor_info))
        avg_bond_len    = float(np.mean(bond_lengths)) if bond_lengths else 0.0

        # --- Vecinos a 2 saltos ---
        neighbors2 = set()
        for nb in neighbors:
            for nb2 in nb.GetNeighbors():
                if nb2.GetIdx() != i:
                    neighbors2.add(nb2.GetIdx())

        neighbor2_elems = [get_element(idx) for idx in neighbors2]
        neighbor2_config = "-".join(sorted(
            [f"{e}" for e in neighbor2_elems]
        ))
        n_electroneg_2 = sum(1 for e in neighbor2_elems if e in ELECTRONEGATIVE)

        # --- Ángulos ---
        angles  = []
        nb_list = list(neighbors)
        for j in range(len(nb_list)):
            for k in range(j + 1, len(nb_list)):
                a_xyz = get_tripos(nb_list[j].GetIdx()).get("xyz", tuple(conf.GetAtomPosition(nb_list[j].GetIdx())))
                c_xyz = get_tripos(nb_list[k].GetIdx()).get("xyz", tuple(conf.GetAtomPosition(nb_list[k].GetIdx())))
                angles.append(angle_deg(a_xyz, xyz_i, c_xyz))
        avg_angle = float(np.mean(angles)) if angles else 0.0

        # --- Planaridad ---
        if tripos_type in PLANAR_TYPES:
            planar = True
        elif coordination >= 3:
            planar = is_planar_by_geometry(xyz_i, neighbor_xyzs)
        else:
            planar = False

        # --- Features de anillo ---
        is_in_ring  = ring_info.NumAtomRings(i) > 0
        atom_rings  = [r for r in ring_info.AtomRings() if i in r]
        ring_size   = min(len(r) for r in atom_rings) if atom_rings else 0
        is_aromatic = atom.GetIsAromatic()

        # --- Carga formal y grado de insaturación ---
        formal_charge = atom.GetFormalCharge()
        degree_of_unsat = bond_type_counts["double"] + bond_type_counts["aromatic"]

        # --- Carga de Gasteiger ---
        if has_gasteiger:
            try:
                gasteiger_charge = float(atom.GetDoubleProp("_GasteigerCharge"))
                if math.isnan(gasteiger_charge) or math.isinf(gasteiger_charge):
                    gasteiger_charge = 0.0
            except Exception:
                gasteiger_charge = 0.0
        else:
            gasteiger_charge = 0.0

        # --- Targets desde .itp ---
        itp_atom = itp_data.get(atom_name)
        if itp_atom is None and itp_data:
            unmatched.append(atom_name)
        atomtype = itp_atom["atomtype"] if itp_atom else None
        charge   = itp_atom["charge"]   if itp_atom else None

        rows.append({
            # Identificadores
            "molecule":               mol_name,
            "atom_id":                i + 1,
            "atom_name":              atom_name,
            # Features originales
            "tripos_type":            tripos_type,
            "element":                element,
            "mass":                   round(mass, 4),
            "coordination":           coordination,
            "neighbor_config":        neighbor_config,
            "n_dihedrals":            dihedral_counts[i],
            "is_planar":              planar,
            "avg_bond_len_A":         round(avg_bond_len, 4),
            "avg_angle_deg":          round(avg_angle, 4),
            "C_count":                elem_counts["C"],
            "H_count":                elem_counts["H"],
            "O_count":                elem_counts["O"],
            "N_count":                elem_counts["N"],
            # Features nuevos — entorno extendido
            "neighbor2_config":       neighbor2_config,
            "n_electroneg_neighbors": n_electroneg_1,
            "n_electroneg_neighbors2": n_electroneg_2,
            # Features nuevos — tipos de enlace
            "bonds_single":           bond_type_counts["single"],
            "bonds_double":           bond_type_counts["double"],
            "bonds_aromatic":         bond_type_counts["aromatic"],
            # Features nuevos — anillo
            "is_in_ring":             is_in_ring,
            "ring_size":              ring_size,
            "is_aromatic":            is_aromatic,
            # Features nuevos — carga y electronegatividad
            "formal_charge":          formal_charge,
            "degree_of_unsat":        degree_of_unsat,
            "gasteiger_charge":       round(gasteiger_charge, 6),
            # Targets
            "atomtype":               atomtype,
            "charge":                 charge,
        })

    if unmatched:
        print(f"  [WARN] {mol2_path.name}: {len(unmatched)} átomo(s) sin match en .itp → {unmatched}")

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Extrae features del .mol2 y targets del .itp en un único CSV"
    )
    parser.add_argument("--input",  "-i", required=True, help="Carpeta con archivos .mol2 e .itp")
    parser.add_argument("--output", "-o", default="dataset.csv", help="Archivo CSV de salida")
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        sys.exit(f"Error: '{input_dir}' no es una carpeta válida.")

    mol2_files = sorted(input_dir.glob("*.mol2"))
    if not mol2_files:
        sys.exit(f"No se encontraron archivos .mol2 en '{input_dir}'.")

    print(f"Encontrados {len(mol2_files)} archivos .mol2 en '{input_dir}'")

    all_rows = []
    skipped  = []

    for mol2_path in mol2_files:
        itp_path = mol2_path.with_suffix(".itp")
        if not itp_path.exists():
            print(f"  [SKIP] Sin .itp para: {mol2_path.name}")
            skipped.append(mol2_path.name)
            continue

        print(f"  Procesando: {mol2_path.name} ...", end=" ")
        rows = extract_molecule(mol2_path, itp_path)
        all_rows.extend(rows)
        print(f"{len(rows)} átomos")

    if not all_rows:
        sys.exit("No se pudo extraer ningún átomo. Verificá los archivos.")

    df = pd.DataFrame(all_rows, columns=[
        "molecule", "atom_id", "atom_name",
        "tripos_type", "element", "mass",
        "coordination", "neighbor_config", "n_dihedrals", "is_planar",
        "avg_bond_len_A", "avg_angle_deg",
        "C_count", "H_count", "O_count", "N_count",
        "neighbor2_config", "n_electroneg_neighbors", "n_electroneg_neighbors2",
        "bonds_single", "bonds_double", "bonds_aromatic",
        "is_in_ring", "ring_size", "is_aromatic",
        "formal_charge", "degree_of_unsat", "gasteiger_charge",
        "atomtype", "charge",
    ])

    output_path = Path(args.output)
    df.to_csv(output_path, index=False)
    print(f"\n✓ CSV guardado en: {output_path}")
    print(f"  {len(df)} filas · {df['molecule'].nunique()} moléculas · {len(df.columns)} columnas")
    if skipped:
        print(f"  Moléculas sin .itp ({len(skipped)}): {skipped}")


if __name__ == "__main__":
    main()
