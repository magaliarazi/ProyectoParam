
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds

# ─────────────────────────────────────────────
# Tabla de masas atómicas (las más comunes en
# moléculas orgánicas / lípidos / proteínas)
# ─────────────────────────────────────────────
ATOMIC_MASS = {
    "H": 1.008,  "C": 12.011, "N": 14.007, "O": 15.999,
    "P": 30.974, "S": 32.06,  "F": 18.998, "Cl": 35.45,
    "Br": 79.904,"I": 126.904,"Na": 22.990,"Mg": 24.305,
    "Ca": 40.078,"Fe": 55.845,"Zn": 65.38,
}

# ─────────────────────────────────────────────
# Helpers geométricos
# ─────────────────────────────────────────────

def get_positions(conf, indices):
    return [np.array(conf.GetAtomPosition(i)) for i in indices]


def calc_avg_bond_length(pos_center, pos_neighbors):
    if not pos_neighbors:
        return 0.0
    return float(np.mean([np.linalg.norm(p - pos_center) for p in pos_neighbors]))


def calc_avg_angle(pos_center, pos_neighbors):
    angles = []
    n = len(pos_neighbors)
    for i in range(n):
        for j in range(i + 1, n):
            v1 = pos_neighbors[i] - pos_center
            v2 = pos_neighbors[j] - pos_center
            norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if norm1 < 1e-8 or norm2 < 1e-8:
                continue
            cos_a = np.dot(v1, v2) / (norm1 * norm2)
            angles.append(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))
    return float(np.mean(angles)) if angles else 0.0


def calc_is_planar(pos_center, pos_neighbors, threshold=0.15):
    """
    Usa SVD sobre las posiciones relativas de los vecinos.
    Si el menor valor singular es < threshold → plano.
    Requiere al menos 3 vecinos.
    """
    if len(pos_neighbors) < 3:
        return False
    pts = np.array(pos_neighbors) - pos_center
    _, s, _ = np.linalg.svd(pts)
    return bool(s[-1] < threshold)


def count_dihedrals(mol, atom_idx):
    """
    Cuenta cuántos ángulos diedros involucran al átomo dado
    (es decir, cuántos pares de vecinos-de-vecinos existen
    a través de este átomo).
    """
    atom = mol.GetAtomWithIdx(atom_idx)
    neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
    count = 0
    for nb_idx in neighbors:
        nb_atom = mol.GetAtomWithIdx(nb_idx)
        # vecinos del vecino, excluyendo el átomo central
        nn = [n.GetIdx() for n in nb_atom.GetNeighbors() if n.GetIdx() != atom_idx]
        count += len(nn)
    return count


# ─────────────────────────────────────────────
# Carga del PDB con inferencia de bonds
# ─────────────────────────────────────────────

def load_pdb(pdb_path: str):
    """
    Carga el PDB y determina la conectividad usando rdDetermineBonds.
    Devuelve el mol con hidrógenos explícitos.
    """
    raw = Chem.MolFromPDBFile(pdb_path, removeHs=False, sanitize=False)
    if raw is None:
        raise ValueError(f"No se pudo leer el archivo: {pdb_path}")

    # Intentar determinar bonds a partir de distancias
    try:
        rdDetermineBonds.DetermineConnectivity(raw)
        rdDetermineBonds.DetermineBondOrders(raw, charge=0)
    except Exception:
        # Si falla el orden de bonds, al menos tenemos conectividad
        pass

    try:
        Chem.SanitizeMol(raw)
    except Exception:
        pass

    return raw


# ─────────────────────────────────────────────
# Extracción de features por átomo
# ─────────────────────────────────────────────

def extract_features(mol, molecule_name: str) -> pd.DataFrame:
    conf = mol.GetConformer()
    rows = []

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        element = atom.GetSymbol()
        mass = ATOMIC_MASS.get(element, atom.GetMass())

        neighbor_indices = [n.GetIdx() for n in atom.GetNeighbors()]
        coordination = len(neighbor_indices)
        neighbor_symbols = tuple(sorted([mol.GetAtomWithIdx(n).GetSymbol()
                                         for n in neighbor_indices]))
        neighbor_config = "-".join(neighbor_symbols) if neighbor_symbols else "none"

        pos_center = np.array(conf.GetAtomPosition(idx))
        pos_neighbors = get_positions(conf, neighbor_indices)

        avg_bond_len = calc_avg_bond_length(pos_center, pos_neighbors)
        avg_angle = calc_avg_angle(pos_center, pos_neighbors)
        is_planar = calc_is_planar(pos_center, pos_neighbors)
        n_dihedrals = count_dihedrals(mol, idx)

        rows.append({
            "molecule":        molecule_name,
            "element":         element,
            "mass":            round(mass, 4),
            "coordination":    coordination,
            "neighbor_config": neighbor_config,
            "n_dihedrals":     n_dihedrals,
            "is_planar":       int(is_planar),
            "avg_bond_len_A":  round(avg_bond_len, 4),
            "avg_angle_deg":   round(avg_angle, 4),
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    pdb_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else Path(pdb_path).stem + ".csv"

    molecule_name = Path(pdb_path).stem

    print(f"Leyendo: {pdb_path}")
    mol = load_pdb(pdb_path)
    print(f"  → {mol.GetNumAtoms()} átomos cargados")

    df = extract_features(mol, molecule_name)
    df.to_csv(out_path, index=False)

    print(f"CSV guardado en: {out_path}")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()