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
            if not linea:
                continue

            if linea.startswith("["):
                seccion_actual = linea.strip("[]").strip().lower()
                continue

            partes = linea.split()
            if not partes:
                continue

            # ---------- molecule ----------
            if seccion_actual == "moleculetype" and not datos["molecule_name"]:
                datos["molecule_name"] = partes[0]

            # ---------- atoms ----------
            elif seccion_actual == "atoms" and len(partes) >= 8:
                try:
                    atom_id = int(partes[0])
                    nombre = partes[4]
                    datos["atoms"].append({
                        "id": atom_id,
                        "atom_type": partes[1],
                        "elemento": obtener_elemento(nombre),
                        "carga": float(partes[6]),
                        "masa": float(partes[7]),
                    })
                except ValueError:
                    pass

            # ---------- bonds ----------
            elif seccion_actual == "bonds" and len(partes) >= 2:
                try:
                    ai, aj = int(partes[0]), int(partes[1])
                    dist_nm = float(partes[3]) if len(partes) > 3 else np.nan
                    dist = dist_nm * 10 if not np.isnan(dist_nm) else np.nan
                    datos["bonds"].append({"ai": ai, "aj": aj, "dist": dist})
                except ValueError:
                    pass

            # ---------- dihedrals (para planaridad) ----------
            elif seccion_actual == "dihedrals" and len(partes) >= 5:
                try:
                    aj = int(partes[1])
                    func = int(partes[4])
                    if func in (2, 4): # GROMOS Impropers
                        datos["impropers"].add(aj)
                except ValueError:
                    pass

    return datos

# =====================================================
# 3. CONSTRUCTOR DE GRAFOS
# =====================================================
def crear_base_datos_grafos(directorio_itp):
    biblioteca_grafos = {}
    archivos = list(Path(directorio_itp).glob("*.itp"))
    
    print(f"📂 Procesando {len(archivos)} archivos para arquitectura de grafos...")

    for archivo in archivos:
        mol = extraer_datos_itp(archivo)
        mol_id = mol["molecule_name"] if mol["molecule_name"] else archivo.stem
        
        # --- NODOS (Características Atómicas) ---
        nodos = {}
        for a in mol["atoms"]:
            nodos[a["id"]] = {
                "elemento": a["elemento"],
                "masa": a["masa"],
                "carga_target": a["carga"],
                "tipo_target": a["atom_type"],
                "es_planar": 1 if a["id"] in mol["impropers"] else 0
            }

        # --- ARISTAS (Conectividad) ---
        aristas = []
        for b in mol["bonds"]:
            aristas.append({
                "indices": (b["ai"], b["aj"]),
                "distancia_A": b["dist"]
            })

        # --- ENSAMBLADO DEL GRAFO ---
        biblioteca_grafos[mol_id] = {
            "nodos": nodos,
            "aristas": aristas,
            "filename": archivo.name
        }
        
        print(f"   ✅ Grafo construido: {mol_id} ({len(nodos)} nodos)")

    return biblioteca_grafos

# =====================================================
# 4. EJECUCIÓN
# =====================================================
if __name__ == "__main__":
    # Definir rutas (ajustadas a tu terminal)
    SCRIPT_DIR = Path(__file__).resolve().parent
    BASE = SCRIPT_DIR.parent
    ruta_itp = BASE / "input" # Asegúrate de que esta carpeta tenga los .itp

    if not ruta_itp.exists():
        print(f"❌ Error: No existe la carpeta {ruta_itp}")
    else:
        # 1. Generar la arquitectura de grafos
        dataset_grafos = crear_base_datos_grafos(ruta_itp)

        # 2. Guardar en formato Pickle
        # Pickle es ideal porque guarda diccionarios y objetos complejos de Python
        output_file = SCRIPT_DIR / "dataset_grafos.pkl"
        with open(output_file, "wb") as f:
            pickle.dump(dataset_grafos, f)

        print(f"\n✨ Proceso finalizado.")
        print(f"💾 Archivo generado: {output_file}")
        print(f"💡 Esta base de datos ahora permite navegación topológica por molécula.")
