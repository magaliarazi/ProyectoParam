import os
import networkx as nx
import pandas as pd


def parse_itp(file_path):
    atoms = {}
    bonds = []
    section = None

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith(";"):
                continue

            if line.startswith("["):
                section = line.lower()
                continue

            # ---------- ATOMS ----------
            if "atoms" in section:
                parts = line.split()
                if len(parts) < 8:
                    continue

                atom_id = int(parts[0])
                atom_type = parts[1]
                charge = float(parts[6])

                element = ''.join([c for c in atom_type if c.isalpha()]).upper()

                atoms[atom_id] = {
                    "type": atom_type,
                    "element": element,
                    "charge": charge
                }

            # ---------- BONDS ----------
            elif "bonds" in section:
                parts = line.split()
                if len(parts) < 2:
                    continue

                ai = int(parts[0])
                aj = int(parts[1])
                bonds.append((ai, aj))

    return atoms, bonds


def build_graph(atoms, bonds):
    G = nx.Graph()

    for atom_id, data in atoms.items():
        G.add_node(atom_id, **data)

    for ai, aj in bonds:
        G.add_edge(ai, aj)

    return G


def generate_features(G, mol_name):
    rows = []

    for node in G.nodes():
        neighbors = list(G.neighbors(node))

        element = G.nodes[node]["element"]
        degree = len(neighbors)

        num_C = sum(1 for n in neighbors if G.nodes[n]["element"] == "C")
        num_O = sum(1 for n in neighbors if G.nodes[n]["element"] == "O")
        num_N = sum(1 for n in neighbors if G.nodes[n]["element"] == "N")
        num_H = sum(1 for n in neighbors if G.nodes[n]["element"] == "H")

        row = {
            "molecule": mol_name,
            "atom_id": node,
            "element": element,
            "degree": degree,
            "neighbors_C": num_C,
            "neighbors_O": num_O,
            "neighbors_N": num_N,
            "neighbors_H": num_H,
            "charge_target": G.nodes[node]["charge"],
            "atom_type_target": G.nodes[node]["type"]
        }

        rows.append(row)

    return pd.DataFrame(rows)


def process_folder(folder_path, output_csv):
    all_data = []

    for file in os.listdir(folder_path):
        if file.endswith(".itp"):
            path = os.path.join(folder_path, file)
            mol_name = file.replace(".itp", "")

            print(f"Procesando: {file}")

            atoms, bonds = parse_itp(path)
            G = build_graph(atoms, bonds)
            df = generate_features(G, mol_name)

            all_data.append(df)

    final_df = pd.concat(all_data, ignore_index=True)
    final_df.to_csv(output_csv, index=False)

    print(f"\nDataset final guardado en: {output_csv}")


# -------- USO --------
if __name__ == "__main__":
    folder = "/home/marazi/proyectoParam/input/.itp/"
    output = "dataset_grafos.csv"

    process_folder(folder, output)
