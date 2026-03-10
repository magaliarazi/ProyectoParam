import os
import pandas as pd
import re

def extraer_datos_itp(ruta_archivo):
    datos = {'atoms': [], 'bonds': [], 'angles': [], 'impropers': [], 'molecule_name': ''}
    seccion_actual = None

    with open(ruta_archivo, 'r') as f:
        for linea in f:
            linea = linea.split(';')[0].strip()  # Ignorar comentarios
            if not linea:
                continue

            if linea.startswith('['):
                seccion_actual = linea.replace('[', '').replace(']', '').strip()
                continue

            partes = linea.split()
            if not partes:
                continue

            # Nombre de la molécula
            if seccion_actual == 'moleculetype' and not datos['molecule_name']:
                datos['molecule_name'] = partes[0]

            # Átomos
            if seccion_actual == 'atoms' and len(partes) >= 8:
                datos['atoms'].append({
                    'id': int(partes[0]),
                    'atom_type': partes[1],
                    'nombre_itp': partes[4],
                    'elemento': re.sub(r'\d+', '', partes[4]),
                    'carga': float(partes[6]),
                    'masa': float(partes[7])
                })

            # Bonds
            elif seccion_actual == 'bonds' and len(partes) >= 2:
                dist = float(partes[3]) if len(partes) > 3 else 0.0
                datos['bonds'].append({'ai': int(partes[0]), 'aj': int(partes[1]), 'dist': dist})

            # Angles
            elif seccion_actual == 'angles' and len(partes) >= 3:
                angle = float(partes[4]) if len(partes) > 4 else 0.0
                datos['angles'].append({'ai': int(partes[0]), 'aj': int(partes[1]), 'ak': int(partes[2]), 'angle': angle})

            # Impropers / dihedrals
            elif seccion_actual == 'dihedrals' and len(partes) >= 2:
                datos['impropers'].append(int(partes[1]))

    return datos

def crear_base_datos_subcarpetas(directorio_principal):
    registros = []
    for subcarpeta in os.listdir(directorio_principal):
        ruta_sub = os.path.join(directorio_principal, subcarpeta)
        if not os.path.isdir(ruta_sub):
            continue

        # Buscar .itp dentro de la subcarpeta
        itp_encontrado = False
        for archivo in os.listdir(ruta_sub):
            if archivo.endswith('.itp'):
                ruta_itp = os.path.join(ruta_sub, archivo)
                mol = extraer_datos_itp(ruta_itp)
                itp_encontrado = True

                # Crear adyacencia y distancias
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
                    avg_dist = sum(dists)/len(dists) if dists else 0
                    angulos_atomo = [ang['angle'] for ang in mol['angles'] if ang['aj'] == id_a]
                    avg_angle = sum(angulos_atomo)/len(angulos_atomo) if angulos_atomo else 0

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

        if not itp_encontrado:
            print(f"⚠️  No se encontró .itp en {subcarpeta}")

    return pd.DataFrame(registros)

# =========================
# Configuración de ruta y salida
ruta_input_principal = '/home/marazi/proyectoParam/input/automatico/nuevasmoleculas/acpype_out'
nombre_salida = 'base_datos_acpype.csv'
# =========================

df = crear_base_datos_subcarpetas(ruta_input_principal)
df.to_csv(nombre_salida, index=False)
print(f"✅ Base generada con {len(df)} ejemplos en '{nombre_salida}'")
