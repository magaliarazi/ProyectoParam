import os
import pandas as pd
import re
import numpy as np

def extraer_datos_itp(ruta_archivo):
    datos = {'atoms': [], 'bonds': [], 'angles': [], 'impropers': [], 'molecule_name': ''}
    seccion_actual = None

    with open(ruta_archivo, 'r') as f:
        for linea in f:
            linea = linea.split(';')[0].strip()
            if not linea:
                continue

            if linea.startswith('['):
                seccion_actual = linea.replace('[', '').replace(']', '').strip().lower()
                continue

            partes = linea.split()
            if not partes:
                continue

            if seccion_actual == 'moleculetype' and not datos['molecule_name']:
                datos['molecule_name'] = partes[0]

            # 1. Datos del átomo
            if seccion_actual == 'atoms' and len(partes) >= 8:
                datos['atoms'].append({
                    'id': int(partes[0]),
                    'atom_type': partes[1],
                    'nombre_itp': partes[4],
                    'elemento': re.sub(r'\d+', '', partes[4]),
                    'carga': float(partes[6]),
                    'masa': float(partes[7])
                })

            # 2. Conectividad (Bonds)
            elif seccion_actual == 'bonds' and len(partes) >= 2:
                bond_dist = float(partes[3]) if len(partes) > 3 else 0.0
                datos['bonds'].append({'ai': int(partes[0]), 'aj': int(partes[1]), 'dist': bond_dist})

            # 3. Geometría (Angles)
            elif seccion_actual == 'angles' and len(partes) >= 3:
                angle_val = float(partes[4]) if len(partes) > 4 else 0.0
                datos['angles'].append({'ai': int(partes[0]), 'aj': int(partes[1]), 'ak': int(partes[2]), 'angle': angle_val})

            # 4. Impropers
            elif seccion_actual == 'dihedrals':
                try:
                    func = int(partes[4]) if len(partes) > 4 else None
                    if func == 4:  # solo impropers
                        datos['impropers'].append(int(partes[1]))
                except:
                    pass

    return datos

def crear_base_datos_recursivo(directorio_input):
    registros = []
    for root, dirs, files in os.walk(directorio_input):
        for archivo in files:
            if archivo.endswith('.itp'):
                ruta_archivo = os.path.join(root, archivo)
                mol = extraer_datos_itp(ruta_archivo)

                ady = {a['id']: [] for a in mol['atoms']}
                distancias = {}
                for b in mol['bonds']:
                    ady[b['ai']].append(b['aj'])
                    ady[b['aj']].append(b['ai'])
                    distancias[tuple(sorted((b['ai'], b['aj'])))] = b['dist']

                for atom in mol['atoms']:
                    id_a = atom['id']
                    vecinos_ids = ady[id_a]
                    vecinos_elem = sorted([next(a['elemento'] for a in mol['atoms'] if a['id'] == v) for v in vecinos_ids])
                    dists = [distancias[tuple(sorted((id_a, v)))] for v in vecinos_ids]
                    avg_dist = sum(dists)/len(dists) if dists else np.nan
                    angulos_atomo = [ang['angle'] for ang in mol['angles'] if ang['aj'] == id_a]
                    avg_angle = sum(angulos_atomo)/len(angulos_atomo) if angulos_atomo else np.nan

                    registros.append({
                        'molecule': mol['molecule_name'],
                        'element': atom['elemento'],
                        'mass': atom['masa'],
                        'coordination': len(vecinos_ids),
                        'neighbor_config': "-".join(vecinos_elem),
                        'avg_bond_len': round(avg_dist, 4),
                        'avg_angle': round(avg_angle, 2),
                        'is_planar': 1 if id_a in mol['impropers'] else 0,
                        'target_atom_type': atom['atom_type'],
                        'target_charge': atom['carga']
                    })

    return pd.DataFrame(registros)

# Cambiar según tu ruta principal
ruta_input = '/home/marazi/proyectoParam/input/automatico/nuevasmoleculas/acpype_out'
df = crear_base_datos_recursivo(ruta_input)
df.to_csv('base_datos_nomodif.csv', index=False)
print(f"Base generada con {len(df)} ejemplos.")
