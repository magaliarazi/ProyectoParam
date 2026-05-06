import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Masas atómicas estándar
ATOMIC_MASSES = {
    "H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999,
    "F": 18.998, "P": 30.974, "S": 32.06, "Cl": 35.45,
    "Br": 79.904, "I": 126.904,
}

PLANAR_TYPES = {"C.2", "C.ar", "N.ar", "N.2", "N.pl3", "O.2", "S.2", "N.am"}
ELECTRONEGATIVE = {"N", "O", "F", "Cl", "S", "Br", "I"}


# =========================
# PARSERS
# =========================

def parse_mol2_atoms(path):
    atoms = {}
    in_block = False

    with open(path) as f:
        for line in f:
            if line.startswith("@<TRIPOS>ATOM"):
                in_block = True
                continue
            if line.startswith("@<TRIPOS>") and in_block:
                break
            if in_block and line.strip():
                p = line.split()
                atoms[int(p[0])] = {
                    "name": p[1],
                    "xyz": (float(p[2]), float(p[3]), float(p[4])),
                    "type": p[5]
                }
    return atoms


def parse_mol2_bonds(path):
    bonds = []
    in_block = False

    with open(path) as f:
        for line in f:
            if line.startswith("@<TRIPOS>BOND"):
                in_block = True
                continue
            if line.startswith("@<TRIPOS>") and in_block:
                break
            if in_block and line.strip():
                p = line.split()
                bonds.append((int(p[1]), int(p[2]), p[3]))
    return bonds


def parse_itp_atoms(path):
    atoms = {}
    in_block = False

    with open(path) as f:
        for line in f:
            line = line.strip()
            if "[ atoms ]" in line:
                in_block = True
                continue
            if in_block and line.startswith("["):
                break
            if not line or line.startswith(";"):
                continue
            if in_block:
                p = line.split()
                atoms[p[4]] = {
                    "atomtype": p[1],
                    "charge": float(p[6])
                }
    return atoms


# =========================
# GEOMETRÍA
# =========================

def dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def angle(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-12)
    return math.degrees(math.acos(np.clip(cosang, -1, 1)))


def is_planar(center, neighs):
    if len(neighs) < 3:
        return True
    pts = np.array([center] + neighs)
    pts -= pts.mean(axis=0)
    _, s, _ = np.linalg.svd(pts)
    return s[-1] < 0.1


# =========================
# MAIN
# =========================

def extract(mol2_path, itp_path):
    atoms = parse_mol2_atoms(mol2_path)
    bonds = parse_mol2_bonds(mol2_path)
    itp = parse_itp_atoms(itp_path)

    # grafo
    neigh = {i: [] for i in atoms}
    bond_types = {}

    for a, b, t in bonds:
        neigh[a].append(b)
        neigh[b].append(a)
        bond_types[(a, b)] = t
        bond_types[(b, a)] = t

    rows = []

    for i, data in atoms.items():
        xyz = data["xyz"]
        ttype = data["type"]
        element = ttype.split(".")[0]

        neighbors = neigh[i]
        coord = len(neighbors)

        neigh_xyz = []
        bond_lens = []
        bond_count = {"single":0,"double":0,"ar":0}

        elem_counts = {"C":0,"H":0,"O":0,"N":0}
        n_elec1 = 0

        for j in neighbors:
            nb = atoms[j]
            nb_elem = nb["type"].split(".")[0]

            if nb_elem in elem_counts:
                elem_counts[nb_elem]+=1
            if nb_elem in ELECTRONEGATIVE:
                n_elec1+=1

            bl = dist(xyz, nb["xyz"])
            bond_lens.append(bl)
            neigh_xyz.append(nb["xyz"])

            bt = bond_types[(i,j)]
            if bt == "1":
                bond_count["single"]+=1
            elif bt == "2":
                bond_count["double"]+=1
            elif bt == "ar":
                bond_count["ar"]+=1

        avg_len = np.mean(bond_lens) if bond_lens else 0

        # vecinos 2
        neigh2 = set()
        for j in neighbors:
            for k in neigh[j]:
                if k != i:
                    neigh2.add(k)

        n_elec2 = sum(
            1 for k in neigh2
            if atoms[k]["type"].split(".")[0] in ELECTRONEGATIVE
        )

        # ángulos
        angles = []
        for a in neighbors:
            for b in neighbors:
                if a < b:
                    angles.append(angle(atoms[a]["xyz"], xyz, atoms[b]["xyz"]))
        avg_ang = np.mean(angles) if angles else 0

        planar = ttype in PLANAR_TYPES or is_planar(xyz, neigh_xyz)

        # targets
        atom_name = data["name"]
        itp_atom = itp.get(atom_name)

        rows.append({
            "molecule": mol2_path.stem,
            "atom_id": i,
            "atom_name": atom_name,
            "element": element,
            "mass": ATOMIC_MASSES.get(element, 0),
            "coordination": coord,
            "avg_bond_len": avg_len,
            "avg_angle": avg_ang,
            "is_planar": planar,
            "C_count": elem_counts["C"],
            "H_count": elem_counts["H"],
            "O_count": elem_counts["O"],
            "N_count": elem_counts["N"],
            "n_electroneg_1": n_elec1,
            "n_electroneg_2": n_elec2,
            "bonds_single": bond_count["single"],
            "bonds_double": bond_count["double"],
            "bonds_ar": bond_count["ar"],
            "atomtype": itp_atom["atomtype"] if itp_atom else None,
            "charge": itp_atom["charge"] if itp_atom else None
        })

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", required=True)
    parser.add_argument("-o", default="dataset.csv")
    args = parser.parse_args()

    path = Path(args.i)
    rows = []

    for mol2 in path.glob("*.mol2"):
        itp = mol2.with_suffix(".itp")
        if not itp.exists():
            continue
        print("Procesando", mol2.name)
        rows.extend(extract(mol2, itp))

    df = pd.DataFrame(rows)
    df.to_csv(args.o, index=False)
    print("✔ dataset listo")


if __name__ == "__main__":
    main()