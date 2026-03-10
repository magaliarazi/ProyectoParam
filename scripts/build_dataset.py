import pandas as pd
from pathlib import Path

INPUT_DIR = Path("../input")
OUTPUT_FILE = "dataset_all_molecules.csv"

all_atoms = []

def parse_itp(itp_path):
    atoms = []
    bonds = []
    section = None

    with open(itp_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";"):
                continue

            if line.startswith("["):
                section = line.lower()
                continue

            if section == "[ atoms ]":
                parts = line.split()
                atoms.append({
                    "atom_id": int(parts[0]),
                    "atom_type": parts[1],
                    "charge": float(parts[6])
                })

            elif section == "[ bonds ]":
                parts = line.split()
                bonds.append((int(parts[0]), int(parts[1])))

    # vecinos
    neighbors = {a["atom_id"]: 0 for a in atoms}
    for a, b in bonds:
        neighbors[a] += 1
        neighbors[b] += 1

    for a in atoms:
        a["n_neighbors"] = neighbors[a["atom_id"]]

    return atoms

# recorrer todos los itp
for itp_file in INPUT_DIR.glob("*.itp"):
    molecule = itp_file.stem
    atoms = parse_itp(itp_file)

    for a in atoms:
        a["molecule"] = molecule
        all_atoms.append(a)

# DataFrame final
df = pd.DataFrame(all_atoms)

df.to_csv(OUTPUT_FILE, index=False)
print(f"Dataset generado: {OUTPUT_FILE}")
print(df.head())
