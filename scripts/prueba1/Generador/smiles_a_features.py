import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# =====================================================
# MASAS ATÓMICAS (igual lógica que tu pipeline)
# =====================================================
ATOM_MASS = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998,
    "Cl": 35.45,
    "S": 32.06,
}


# =====================================================
# FEATURES AUXILIARES
# =====================================================
def get_element(atom):
    return atom.GetSymbol()


def get_mass(atom):
    return ATOM_MASS.get(atom.GetSymbol(), 0.0)


def coordination(atom):
    return len(atom.GetNeighbors())


def neighbor_config(atom):
    neigh = [n.GetSymbol() for n in atom.GetNeighbors()]
    return "-".join(sorted(neigh))


def is_planar(atom):
    return 1 if atom.GetHybridization().name == "SP2" else 0


def count_dihedrals(atom, mol):
    idx = atom.GetIdx()
    count = 0

    for bond1 in atom.GetBonds():
        a2 = bond1.GetOtherAtomIdx(idx)

        for bond2 in mol.GetAtomWithIdx(a2).GetBonds():
            a3 = bond2.GetOtherAtomIdx(a2)
            if a3 != idx:
                count += 1

    return count


def avg_bond_length(atom, conf):
    idx = atom.GetIdx()
    dists = []

    for n in atom.GetNeighbors():
        dist = np.linalg.norm(
            np.array(conf.GetAtomPosition(idx)) -
            np.array(conf.GetAtomPosition(n.GetIdx()))
        )
        dists.append(dist)

    return np.mean(dists) if dists else np.nan


def avg_angle(atom, mol, conf):
    idx = atom.GetIdx()
    neigh = atom.GetNeighbors()

    if len(neigh) < 2:
        return np.nan

    angles = []

    for i in range(len(neigh)):
        for j in range(i + 1, len(neigh)):

            a = neigh[i].GetIdx()
            b = idx
            c = neigh[j].GetIdx()

            v1 = np.array(conf.GetAtomPosition(a)) - np.array(conf.GetAtomPosition(b))
            v2 = np.array(conf.GetAtomPosition(c)) - np.array(conf.GetAtomPosition(b))

            cos_theta = np.dot(v1, v2) / (
                np.linalg.norm(v1) * np.linalg.norm(v2)
            )

            angle = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
            angles.append(angle)

    return np.mean(angles)


# =====================================================
# FUNCIÓN PRINCIPAL
# =====================================================
def smiles_to_features(molecule_name, smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    # 3D para geometría
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    conf = mol.GetConformer()

    rows = []

    for atom in mol.GetAtoms():
        rows.append({
            "molecule": molecule_name,
            "element": get_element(atom),
            "mass": get_mass(atom),
            "coordination": coordination(atom),
            "neighbor_config": neighbor_config(atom),
            "n_dihedrals": count_dihedrals(atom, mol),
            "is_planar": is_planar(atom),
            "avg_bond_len_A": avg_bond_length(atom, conf),
            "avg_angle_deg": avg_angle(atom, mol, conf),
        })

    return pd.DataFrame(rows)


# =====================================================
# EJEMPLO DE USO
# =====================================================
if __name__ == "__main__":

    data = [
        ("mol1", "CCO"),
        ("mol2", "c1ccccc1"),
        ("mol3", "CCN"),
    ]

    dfs = []

    for name, smi in data:
        df = smiles_to_features(name, smi)
        dfs.append(df)

    final_df = pd.concat(dfs, ignore_index=True)

    print(final_df.head())
    final_df.to_csv("features_from_smiles.csv", index=False)