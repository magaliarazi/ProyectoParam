import os
import re
import csv

# -----------------------------
# UTIL
# -----------------------------
def get_element_from_atom_name(atom_name):
    match = re.match(r'^([A-Z][a-z]?)', atom_name)
    return match.group(1) if match else 'X'


# -----------------------------
# PARSER
# -----------------------------
def parse_itp(filepath):
    atoms = {}
    bonds = []
    angles = []
    current_section = None

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(';'):
                continue

            if line.startswith('['):
                current_section = line.strip('[ ]').strip()
                continue

            # ---------- ATOMS ----------
            if current_section == 'atoms':
                parts = line.split()
                if len(parts) >= 8:
                    nr = int(parts[0])
                    atom_name = parts[4]

                    atoms[nr] = {
                        'type': parts[1],
                        'atom': atom_name,
                        'charge': float(parts[6]),
                        'mass': float(parts[7]),
                        'element': get_element_from_atom_name(atom_name),
                    }

            # ---------- BONDS ----------
            elif current_section == 'bonds':
                parts = line.split()
                if len(parts) >= 2:
                    bonds.append((int(parts[0]), int(parts[1])))

            # ---------- ANGLES ----------
            elif current_section == 'angles':
                parts = line.split()
                if len(parts) >= 6:
                    angles.append({
                        'ai': int(parts[0]),
                        'aj': int(parts[1]),
                        'ak': int(parts[2]),
                        'funct': int(parts[3]),
                        'angle': float(parts[4]),
                        'fc': float(parts[5]),
                    })

    return atoms, bonds, angles


# -----------------------------
# FEATURES
# -----------------------------
def compute_coordination(atoms, bonds):
    coord = {nr: 0 for nr in atoms}
    for ai, aj in bonds:
        coord[ai] += 1
        coord[aj] += 1
    return coord


def compute_neighbor_counts(atoms, bonds):
    neighbors = {nr: [] for nr in atoms}

    for ai, aj in bonds:
        neighbors[ai].append(atoms[aj]['element'])
        neighbors[aj].append(atoms[ai]['element'])

    counts = {}
    for nr, neigh_list in neighbors.items():
        counts[nr] = {
            'n_C': neigh_list.count('C'),
            'n_H': neigh_list.count('H'),
            'n_O': neigh_list.count('O'),
            'n_N': neigh_list.count('N'),
            'n_S': neigh_list.count('S'),
        }

    return counts


def build_angle_features(molecule_name, atoms, bonds, angles):
    coord = compute_coordination(atoms, bonds)
    neigh_counts = compute_neighbor_counts(atoms, bonds)

    rows = []

    for ang in angles:
        ai, aj, ak = ang['ai'], ang['aj'], ang['ak']

        a_i = atoms[ai]
        a_j = atoms[aj]
        a_k = atoms[ak]

        row = {
            'molecule': molecule_name,

            # estructura
            'ai': ai,
            'aj': aj,
            'ak': ak,

            # elementos
            'element_i': a_i['element'],
            'element_j': a_j['element'],
            'element_k': a_k['element'],

            # tipos
            'type_i': a_i['type'],
            'type_j': a_j['type'],
            'type_k': a_k['type'],

            # cargas
            'charge_i': a_i['charge'],
            'charge_j': a_j['charge'],
            'charge_k': a_k['charge'],

            # masas
            'mass_i': a_i['mass'],
            'mass_j': a_j['mass'],
            'mass_k': a_k['mass'],

            # coordinación (clave)
            'degree_j': coord[aj],

            # entorno del átomo central (CLAVE)
            'n_C_j': neigh_counts[aj]['n_C'],
            'n_H_j': neigh_counts[aj]['n_H'],
            'n_O_j': neigh_counts[aj]['n_O'],
            'n_N_j': neigh_counts[aj]['n_N'],
            'n_S_j': neigh_counts[aj]['n_S'],

            # targets
            'target_angle': ang['angle'],
            'target_fc': ang['fc'],
        }

        rows.append(row)

    return rows


# -----------------------------
# MAIN
# -----------------------------
def process_folder(folder_path, output_csv):
    all_rows = []

    for filename in os.listdir(folder_path):
        if not filename.endswith('.itp'):
            continue

        filepath = os.path.join(folder_path, filename)

        # -------- molecule name (igual que tu script) --------
        molecule = None
        current = None

        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(';'):
                    continue

                if line.startswith('['):
                    current = line.strip('[ ]').strip()
                    continue

                if current == 'moleculetype':
                    molecule = line.split()[0]
                    break

        if molecule is None:
            molecule = filename.replace('.itp', '')
            print(f"⚠️ No moleculetype en {filename}, usando filename")

        atoms, bonds, angles = parse_itp(filepath)

        rows = build_angle_features(molecule, atoms, bonds, angles)
        all_rows.extend(rows)

        print(f"✓ {molecule}: {len(angles)} angles")

    # guardar
    if all_rows:
        fieldnames = all_rows[0].keys()

        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

        print(f"\n✅ CSV generado: {output_csv} ({len(all_rows)} filas)")


# -----------------------------
# RUN
# -----------------------------
folder = '/home/marazi/proyectoParam/input/.itp'
output = 'angles_dataset.csv'

process_folder(folder, output)