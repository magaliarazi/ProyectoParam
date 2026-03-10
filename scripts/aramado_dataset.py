import os
import pandas as pd
import re

def extraer_datos_itp(ruta_archivo):
    datos = {'atoms': [], 'bonds': [], 'angles': [], 'impropers': [], 'molecule_name': ''}
    seccion_actual = None
    
    with open(ruta_archivo, 'r') as f:
        for linea in f:
            linea = linea.split(';')[0].strip()
            if not linea: continue
            
            if linea.startswith('['):
                seccion_actual = linea.replace('[', '').replace(']', '').strip()
                continue
            
            partes = linea.split()
            
            if seccion_actual == 'moleculetype' and not datos['molecule_name']:
                datos['molecule_name'] = partes[0]
            
            # 1. Datos del Átomo (Target y Masa)
            if seccion_actual == 'atoms' and len(partes) >= 8:
                datos['atoms'].append({
                    'id': int(partes[0]),
                    'atom_type': partes[1], # TARGET
                    'nombre_itp': partes[4],
                    'elemento': re.sub(r'\d+', '', partes[4]), 
                    'carga': float(partes[6]),
                    'masa': float(partes[7])
                })
            
            # 2. Conectividad (Bonds)
            elif seccion_actual == 'bonds' and len(partes) >= 3:
                datos['bonds'].append({'ai': int(partes[0]), 'aj': int(partes[1]), 'dist': float(partes[3])})
            
            # 3. Geometría (Angles)
            elif seccion_actual == 'angles' and len(partes) >= 5:
                datos['angles'].append({'ai': int(partes[0]), 'aj': int(partes[1]), 'ak': int(partes[2]), 'angle': float(partes[4])})
            
            # 4. Rigidez (Impropers)
            elif seccion_actual == 'dihedrals' and len(partes) >= 6:
                datos['impropers'].append(int(partes[1])) 

    return datos

def crear_base_datos(directorio_input):
    registros = []
    for archivo in os.listdir(directorio_input):
        if archivo.endswith('.itp'):
            mol = extraer_datos_itp(os.path.join(directorio_input, archivo))
            
            # Mapa de adyacencia y distancias
            ady = {a['id']: [] for a in mol['atoms']}
            distancias = {}
            for b in mol['bonds']:
                ady[b['ai']].append(b['aj'])
                ady[b['aj']].append(b['ai'])
                distancias[tuple(sorted((b['ai'], b['aj'])))] = b['dist']
            
            for atom in mol['atoms']:
                id_a = atom['id']
                vecinos_ids = ady[id_a]
                
                # Feature 1: Elementos vecinos (ordenados)
                vecinos_elem = sorted([next(a['elemento'] for a in mol['atoms'] if a['id'] == v) for v in vecinos_ids])
                
                # Feature 2: Promedio de distancias de enlace
                dists = [distancias[tuple(sorted((id_a, v)))] for v in vecinos_ids]
                avg_dist = sum(dists)/len(dists) if dists else 0
                
                # Feature 3: Ángulo promedio donde el átomo es el centro
                angulos_atomo = [ang['angle'] for ang in mol['angles'] if ang['aj'] == id_a]
                avg_angle = sum(angulos_atomo)/len(angulos_atomo) if angulos_atomo else 0
                
                registros.append({
                    'molecule': mol['molecule_name'],
                    'element': atom['elemento'],              # Feature 1
                    'mass': atom['masa'],                    # Feature 2
                    'coordination': len(vecinos_ids),         # Feature 3
                    'neighbor_config': "-".join(vecinos_elem), # Feature 4
                    'avg_bond_len': round(avg_dist, 4),       # Feature 5
                    'avg_angle': round(avg_angle, 2),         # Feature 6
                    'is_planar': 1 if id_a in mol['impropers'] else 0, # Feature 7
                    'target_atom_type': atom['atom_type'],    # TARGET 1
                    'target_charge': atom['carga']            # TARGET 2 (opcional)
                })
                
    return pd.DataFrame(registros)

ruta_input = '/home/marazi/proyectoParam/input' 
df = crear_base_datos(ruta_input)
df.to_csv('base_datos_completa.csv', index=False)
print(f"Base generada con {len(df)} ejemplos.")
