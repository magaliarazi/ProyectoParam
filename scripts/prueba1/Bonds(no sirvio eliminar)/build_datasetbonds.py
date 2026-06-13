import os
import pandas as pd
import itertools
from pathlib import Path


INPUT_DIR = Path("/home/marazi/proyectoParam/input/.itp")
OUTPUT_FILE = "bond_dataset.csv"


# -----------------------------
# EXTRAER NOMBRE MOLECULA
# -----------------------------
def extraer_nombre_molecula(filepath):
    seccion = None

    with open(filepath, "r") as f:
        for linea in f:
            linea = linea.split(";")[0].strip()

            if not linea:
                continue

            if linea.startswith("["):
                seccion = linea.strip("[]").strip().lower()
                continue

            if seccion == "moleculetype":
                return linea.split()[0]

    return None


# -----------------------------
# PARSER ITP SIMPLE
# -----------------------------
def parse_itp(filepath):
    atoms = {}
    bonds = []
    section = None

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith(";"):
                continue

            if line.startswith("["):
                if "atoms" in line:
                    section = "atoms"
                elif "bonds" in line:
                    section = "bonds"
                else:
                    section = None
                continue

            parts = line.split()

            if section == "atoms":
                if len(parts) < 2:
                    continue
                atom_id = int(parts[0])
                atom_type = parts[1]
                atoms[atom_id] = {"type": atom_type}

            elif section == "bonds":
                if len(parts) < 2:
                    continue
                i, j = int(parts[0]), int(parts[1])
                bonds.append((i, j))

    return atoms, bonds


# -----------------------------
# DATASET DE ARISTAS
# -----------------------------
def build_edge_dataset(atoms, bonds, molecule_name):
    bond_set = set()
    for i, j in bonds:
        bond_set.add((i, j))
        bond_set.add((j, i))

    atom_ids = list(atoms.keys())
    rows = []

    for i, j in itertools.combinations(atom_ids, 2):
        label = 1 if (i, j) in bond_set else 0

        rows.append({
            "molecule": molecule_name,
            "atom_i": i,
            "atom_j": j,
            "atom_i_type": atoms[i]["type"],
            "atom_j_type": atoms[j]["type"],
            "bond_exists": label
        })

    return rows


# -----------------------------
# MAIN
# -----------------------------
def main():
    all_rows = []

    files = list(INPUT_DIR.glob("*.itp"))

    print(f"📂 Archivos encontrados: {len(files)}")

    for path in files:
        molecule_name = extraer_nombre_molecula(path)

        atoms, bonds = parse_itp(path)

        rows = build_edge_dataset(atoms, bonds, molecule_name)
        all_rows.extend(rows)

        print(f"✔ Procesado {path.name} | atoms={len(atoms)} bonds={len(bonds)}")

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n✅ Dataset guardado en {OUTPUT_FILE}")


if __name__ == "__main__":
    main()