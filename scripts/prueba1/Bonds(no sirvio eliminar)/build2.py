import os
import itertools
import numpy as np
import pandas as pd

INPUT_DIR = "/home/marazi/proyectoParam/input/.itp"
OUTPUT_CSV = "dataset_bonds2.csv"

# =========================
# VALENCIAS SIMPLES
# =========================
VALENCE = {
    "H": 1,
    "C": 4,
    "O": 2,
    "N": 3,
    "S": 2
}

# =========================
# INFERIR ELEMENTO
# =========================
def infer_element(atom_name):
    return ''.join([c for c in atom_name if c.isalpha()])[0]


# =========================
# PARSER ITP
# =========================
def parse_itp(filepath):
    atoms = {}
    bonds = []
    section = None

    with open(filepath, "r") as f:
        for line in f:
            line = line.split(";")[0].strip()
            if not line:
                continue

            if line.startswith("["):
                section = line.strip("[]").strip().lower()
                continue

            parts = line.split()

            if section == "atoms":
                atom_id = int(parts[0])
                atom_type = parts[1]
                atom_name = parts[4]
                charge = float(parts[6])
                mass = float(parts[7])

                element = infer_element(atom_name)

                atoms[atom_id] = {
                    "type": atom_type,
                    "element": element,
                    "charge": charge,
                    "mass": mass
                }

            elif section == "bonds":
                i, j = int(parts[0]), int(parts[1])
                bonds.append((i, j))

    return atoms, bonds


# =========================
# DEGREE
# =========================
def compute_degree(atoms, bonds):
    degree = {k: 0 for k in atoms.keys()}

    for i, j in bonds:
        degree[i] += 1
        degree[j] += 1

    return degree


# =========================
# DATASET
# =========================
def build_dataset(atoms, bonds):

    bond_set = set()
    for i, j in bonds:
        bond_set.add((i, j))
        bond_set.add((j, i))

    degree = compute_degree(atoms, bonds)

    atom_ids = list(atoms.keys())
    rows = []

    for i, j in itertools.combinations(atom_ids, 2):

        a_i = atoms[i]
        a_j = atoms[j]

        label = 1 if (i, j) in bond_set else 0

        val_i = VALENCE.get(a_i["element"], 4)
        val_j = VALENCE.get(a_j["element"], 4)

        rows.append({
            "type_i": a_i["type"],
            "type_j": a_j["type"],
            "element_i": a_i["element"],
            "element_j": a_j["element"],
            "charge_i": a_i["charge"],
            "charge_j": a_j["charge"],
            "mass_i": a_i["mass"],
            "mass_j": a_j["mass"],
            "charge_diff": abs(a_i["charge"] - a_j["charge"]),
            "degree_i": degree[i],
            "degree_j": degree[j],
            "valence_i": val_i,
            "valence_j": val_j,
            "same_element": int(a_i["element"] == a_j["element"]),
            "bond_exists": label
        })

    return rows


# =========================
# BALANCEO
# =========================
def balance_dataset(df, ratio=3):
    pos = df[df["bond_exists"] == 1]
    neg = df[df["bond_exists"] == 0]

    neg_sample = neg.sample(min(len(neg), len(pos) * ratio), random_state=42)

    return pd.concat([pos, neg_sample]).sample(frac=1).reset_index(drop=True)


# =========================
# MAIN
# =========================
def main():

    all_rows = []

    for file in os.listdir(INPUT_DIR):
        if not file.endswith(".itp"):
            continue

        path = os.path.join(INPUT_DIR, file)

        try:
            atoms, bonds = parse_itp(path)
            rows = build_dataset(atoms, bonds)
            all_rows.extend(rows)

            print(f"{file} → atoms={len(atoms)} bonds={len(bonds)}")

        except Exception as e:
            print(f"Error en {file}: {e}")

    df = pd.DataFrame(all_rows)

    print("\nBalanceando dataset...")
    df = balance_dataset(df, ratio=3)

    print("Guardando CSV...")
    df.to_csv(OUTPUT_CSV, index=False)

    print("\n✅ Dataset listo:", OUTPUT_CSV)
    print(df.head())


if __name__ == "__main__":
    main()