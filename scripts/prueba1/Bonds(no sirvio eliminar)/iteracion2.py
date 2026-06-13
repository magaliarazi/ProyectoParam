import os
import json
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


# =====================================================
# CONFIG
# =====================================================
INPUT_DIR = "/home/marazi/proyectoParam/input/.itp"
MAP_PATH = "/home/marazi/proyectoParam/scripts/prueba1/Preprocesamiento/maps_path1/label_maps.json"


# =====================================================
# LOAD MAPS
# =====================================================
with open(MAP_PATH, "r") as f:
    label_maps = json.load(f)


# =====================================================
# PARSER ITP
# =====================================================
def parse_itp(filepath):
    atoms = {}
    bonds = []
    molecule_name = None

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

            if section == "moleculetype" and molecule_name is None:
                molecule_name = parts[0]

            elif section == "atoms":
                atom_id = int(parts[0])
                atom_type = parts[1]
                element = parts[4] if len(parts) > 4 else "C"

                atoms[atom_id] = {
                    "type": atom_type,
                    "element": element
                }

            elif section == "bonds":
                i, j = int(parts[0]), int(parts[1])
                bonds.append((i, j))

    return molecule_name, atoms, bonds


# =====================================================
# DATASET CREATION
# =====================================================
def build_dataset(molecule_name, atoms, bonds):
    bond_set = set()
    for i, j in bonds:
        bond_set.add((i, j))
        bond_set.add((j, i))

    atom_ids = list(atoms.keys())
    rows = []

    # SOLO pares plausibles (reduce ruido extremo)
    for i, j in itertools.combinations(atom_ids, 2):

        # heurística simple química (evita caos)
        if abs(i - j) > 6:
            continue

        label = 1 if (i, j) in bond_set else 0

        rows.append({
            "molecule": molecule_name,
            "atom_i": i,
            "atom_j": j,
            "atom_i_type": atoms[i]["type"],
            "atom_j_type": atoms[j]["type"],
            "element_i": atoms[i]["element"],
            "element_j": atoms[j]["element"],
            "bond_exists": label
        })

    return rows


# =====================================================
# BALANCE
# =====================================================
def balance(df):
    pos = df[df["bond_exists"] == 1]
    neg = df[df["bond_exists"] == 0].sample(len(pos) * 5, random_state=42)

    return pd.concat([pos, neg]).sample(frac=1).reset_index(drop=True)


# =====================================================
# ENCODING
# =====================================================
def encode(df):

    df["molecule"] = df["molecule"].map(label_maps["molecule"])
    df["element_i"] = df["element_i"].map(label_maps["element"])
    df["element_j"] = df["element_j"].map(label_maps["element"])

    df["atom_i_type"] = df["atom_i_type"].map(label_maps["target_atom_type"])
    df["atom_j_type"] = df["atom_j_type"].map(label_maps["target_atom_type"])

    df = df.fillna(0)

    return df


# =====================================================
# MAIN PIPELINE
# =====================================================
def main():

    all_rows = []

    for file in os.listdir(INPUT_DIR):
        if not file.endswith(".itp"):
            continue

        path = os.path.join(INPUT_DIR, file)

        mol_name, atoms, bonds = parse_itp(path)

        rows = build_dataset(mol_name, atoms, bonds)
        all_rows.extend(rows)

        print(f"{file}: atoms={len(atoms)} bonds={len(bonds)}")

    df = pd.DataFrame(all_rows)

    print("\nBalanceando dataset...")
    df = balance(df)

    print("Encoding...")
    df = encode(df)

    # =================================================
    # MODEL
    # =================================================
    features = [
        "molecule",
        "atom_i",
        "atom_j",
        "atom_i_type",
        "atom_j_type",
        "element_i",
        "element_j"
    ]

    X = df[features]
    y = df["bond_exists"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # =================================================
    # METRICS
    # =================================================
    print("\n📊 MATRIZ DE CONFUSIÓN")
    print(confusion_matrix(y_test, y_pred))

    print("\n📊 REPORT")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()