import os
import re
import csv

def get_element_from_atom_name(atom_name):
    """Extrae el elemento del nombre del átomo (ej: C4→C, OH→O, H1→H, Cl2→Cl)"""
    match = re.match(r'^([A-Z][a-z]?)', atom_name)
    return match.group(1) if match else 'X'

def parse_itp(filepath):
    """Parsea un .itp y retorna atoms y bonds como listas de dicts"""
    atoms = {}
    bonds = []
    current_section = None
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(';'):
                continue
            if line.startswith('['):
                current_section = line.strip('[ ]').strip()
                continue
            
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
                        'element': get_element_from_atom_name(atom_name),  # ← directo del nombre
                    }

            elif current_section == 'bonds':
                parts = line.split()
                if len(parts) >= 5:
                    bonds.append({
                        'ai': int(parts[0]),
                        'aj': int(parts[1]),
                        'funct': int(parts[2]),
                        'c0': float(parts[3]),
                        'c1': float(parts[4]),
                    })
    
    return atoms, bonds


def compute_coordination(atoms, bonds):
    coord = {nr: 0 for nr in atoms}
    for b in bonds:
        coord[b['ai']] += 1
        coord[b['aj']] += 1
    return coord


def build_bond_features(molecule_name, atoms, bonds):
    coord = compute_coordination(atoms, bonds)
    rows = []
    
    for b in bonds:
        ai = b['ai']
        aj = b['aj']
        a_i = atoms[ai]
        a_j = atoms[aj]
        
        row = {
            'molecule': molecule_name,
            'ai': ai,
            'aj': aj,
            'element_i': a_i['element'],
            'type_i': a_i['type'],
            'charge_i': a_i['charge'],
            'mass_i': a_i['mass'],
            'coord_i': coord[ai],
            'element_j': a_j['element'],
            'type_j': a_j['type'],
            'charge_j': a_j['charge'],
            'mass_j': a_j['mass'],
            'coord_j': coord[aj],
            'delta_charge': abs(a_i['charge'] - a_j['charge']),
            'delta_mass': abs(a_i['mass'] - a_j['mass']),
            'funct': b['funct'],
            'target_c0': b['c0'],
            'target_c1': b['c1'],
        }
        rows.append(row)
    
    return rows


def process_folder(folder_path, output_csv):
    all_rows = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.itp'):
            filepath = os.path.join(folder_path, filename)
            
            # Leer nombre de molécula desde [moleculetype]
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
                print(f"⚠️  No se encontró moleculetype en {filename}, usando nombre de archivo")
            
            atoms, bonds = parse_itp(filepath)
            rows = build_bond_features(molecule, atoms, bonds)
            all_rows.extend(rows)
            print(f"✓ {molecule}: {len(bonds)} bonds procesados")
    
    if all_rows:
        fieldnames = all_rows[0].keys()
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\n✅ CSV generado: {output_csv} ({len(all_rows)} filas)")


# Ejecutar
folder = '/home/marazi/proyectoParam/input/.itp'
output = 'bonds_dataset.csv'
process_folder(folder, output)