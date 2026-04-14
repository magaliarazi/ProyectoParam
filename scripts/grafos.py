import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# =====================================================
# 1. UTILIDADES QUÍMICAS
# =====================================================
def obtener_elemento(atom_name):
    atom_name = atom_name.strip()
    if atom_name.startswith(("Cl", "Br")):
        return atom_name[:2]
    if atom_name and atom_name[0].isalpha():
        return atom_name[0]
    return atom_name

# =====================================================
# 2. PARSER ITP (Extracción de Estructura)
# =====================================================
def extraer_datos_itp(ruta_archivo):
    datos = {
        "atoms": [],
        "bonds": [],
        "angles": [],
        "dihedrals": [],
        "impropers": set(),
        "molecule_name": "",
    }

    seccion_actual = None

    with open(ruta_archivo, "r") as f:
        for linea in f:
            linea = linea.split(";")[0].strip()
            if not linea: continue

            if linea.startswith("["):
                seccion_actual = linea.strip("[]").strip().lower()
                continue

            partes = linea.split()
            if not partes: continue

            if seccion_actual == "moleculetype" and not datos["molecule_name"]:
                datos["molecule_name"] = partes[0]

            elif seccion_actual == "atoms" and len(partes) >= 8:
                try:
                    datos["atoms"].append({
                        "id": int(partes[0]),
                        "atom_type": partes[1],
                        "elemento": obtener_elemento(partes[4]),
                        "carga": float(partes[6]),
                        "masa": float(partes[7]),
                    })
                except ValueError: pass

            elif seccion_actual == "bonds" and len(partes) >= 2:
                try:
                    ai, aj = int(partes[0]), int(partes[1])
                    dist_nm = float(partes[3]) if len(partes) > 3 else np.nan
                    dist = dist_nm * 10 if not np.isnan(dist_nm) else np.nan
                    datos["bonds"].append({"ai": ai, "aj": aj, "dist": dist})
                except ValueError: pass

            elif seccion_actual == "dihedrals" and len(partes) >= 5:
                try:
                    aj = int(partes[1])
                    func = int(partes[4])
                    if func in (2, 4): datos["impropers"].add(aj)
                except ValueError: pass

    return datos

# =====================================================
# 3. CONSTRUCTOR DE GRAFOS
# =====================================================
def crear_base_datos_grafos(lista_archivos):
    biblioteca_grafos = {}
    
    print(f"📂 Procesando {len(lista_archivos)} archivos para arquitectura de grafos...")

    for archivo in lista_archivos:
        mol = extraer_datos_itp(archivo)
        mol_id = mol["molecule_name"] if mol["molecule_name"] else archivo.stem
        
        nodos = {}
        for a in mol["atoms"]:
            nodos[a["id"]] = {
                "elemento": a["elemento"],
                "masa": a["masa"],
                "carga_target": a["carga"],
                "tipo_target": a["atom_type"],
                "es_planar": 1 if a["id"] in mol["impropers"] else 0
            }

        aristas = []
        for b in mol["bonds"]:
            aristas.append({
                "indices": (b["ai"], b["aj"]),
                "distancia_A": b["dist"]
            })

        biblioteca_grafos[mol_id] = {
            "nodos": nodos,
            "aristas": aristas,
            "filename": archivo.name
        }
        print(f"   ✅ Grafo construido: {mol_id}")

    return biblioteca_grafos

# =====================================================
# 4. EJECUCIÓN (Igual a tu script original)
# =====================================================
if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent
    BASE = SCRIPT_DIR.parent
    
    # Esta es la ruta exacta de tu script original que causaba el conflicto
    # La corregimos para que busque los ARCHIVOS dentro de esa ruta
    ruta_busqueda = BASE / "input" / ".itp"

    # Buscamos todos los archivos .itp dentro de esa ubicación
    # El if f.is_file() evita el error "IsADirectoryError"
    archivos_itp = [f for f in ruta_busqueda.glob("*.itp") if f.is_file()]

    if not archivos_itp:
        print(f"⚠️ No se encontraron archivos .itp en {ruta_busqueda}")
    else:
        dataset_grafos = crear_base_datos_grafos(archivos_itp)

        output_file = SCRIPT_DIR / "dataset_grafos.pkl"
        with open(output_file, "wb") as f:
            pickle.dump(dataset_grafos, f)

        print(f"\n✨ Proceso finalizado.")
        print(f"💾 Archivo de grafos guardado en: {output_file}")
