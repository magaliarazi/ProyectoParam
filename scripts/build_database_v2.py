import pandas as pd
import numpy as np
from pathlib import Path


# =====================================================
# ELEMENTO QUÍMICO REAL
# =====================================================
def obtener_elemento(atom_name):
    atom_name = atom_name.strip()

    if atom_name.startswith(("Cl", "Br")):
        return atom_name[:2]

    if atom_name and atom_name[0].isalpha():
        return atom_name[0]

    return atom_name


# =====================================================
# PARSER ITP
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
                    datos["atoms"].append(
                        {
                            "id": atom_id,
                            "atom_type": partes[1],
                            "elemento": obtener_elemento(nombre),
                            "carga": float(partes[6]),
                            "masa": float(partes[7]),
                        }
                    )
                except ValueError as e:
                    print(f"[DESCARTADA - ValueError] -> {partes} | Error: {e}")

            # ---------- bonds ----------
            elif seccion_actual == "bonds" and len(partes) >= 2:
                try:
                    ai, aj = int(partes[0]), int(partes[1])
                    dist_nm = float(partes[3]) if len(partes) > 3 else np.nan
                    dist    = dist_nm * 10 if not np.isnan(dist_nm) else np.nan
                    # constante de fuerza del bond (c1)
                    fc_bond = float(partes[4]) if len(partes) > 4 else np.nan
                    datos["bonds"].append({
                        "ai": ai, "aj": aj,
                        "dist": dist, "fc_bond": fc_bond
                    })
                except ValueError:
                    pass

            # ---------- angles ----------
            elif seccion_actual == "angles" and len(partes) >= 3:
                try:
                    ai, aj, ak = int(partes[0]), int(partes[1]), int(partes[2])
                    ang    = float(partes[4]) if len(partes) > 4 else np.nan
                    # constante de fuerza del ángulo (fc)
                    fc_ang = float(partes[5]) if len(partes) > 5 else np.nan
                    datos["angles"].append({
                        "ai": ai, "aj": aj, "ak": ak,
                        "angle": ang, "fc_angle": fc_ang
                    })
                except ValueError:
                    pass

            # ---------- dihedrals ----------
            elif seccion_actual == "dihedrals" and len(partes) >= 5:
                try:
                    ai, aj, ak, al = map(int, partes[:4])
                    func = int(partes[4])
                    datos["dihedrals"].append(
                        {"ai": ai, "aj": aj, "ak": ak, "al": al, "func": func}
                    )
                    if func in (2, 4):
                        datos["impropers"].add(aj)
                except ValueError:
                    pass

    return datos


# =====================================================
# DATASET
# =====================================================
def crear_base_datos(directorio_itp):
    registros = []

    archivos = list(Path(directorio_itp).glob("*.itp"))
    print(f"📂 Encontrados {len(archivos)} archivos .itp")

    for archivo in archivos:
        print(f"🔍 Procesando {archivo.name}")
        mol = extraer_datos_itp(archivo)

        atom_map = {a["id"]: a for a in mol["atoms"]}

        # adyacencia
        ady = {a["id"]: [] for a in mol["atoms"]}
        distancias  = {}
        fc_bonds    = {}   # constantes de fuerza de bonds

        for b in mol["bonds"]:
            ady[b["ai"]].append(b["aj"])
            ady[b["aj"]].append(b["ai"])
            key = tuple(sorted((b["ai"], b["aj"])))
            distancias[key] = b["dist"]
            fc_bonds[key]   = b["fc_bond"]

        # contar dihedrales por átomo
        dih_count = {a["id"]: 0 for a in mol["atoms"]}
        for d in mol["dihedrals"]:
            for idx in (d["ai"], d["aj"], d["ak"], d["al"]):
                if idx in dih_count:
                    dih_count[idx] += 1

        # ---------- features por átomo ----------
        for atom in mol["atoms"]:
            id_a       = atom["id"]
            vecinos_ids = ady[id_a]
            vecinos_elem = sorted([atom_map[v]["elemento"] for v in vecinos_ids])

            # longitudes de enlace
            dists = [
                distancias.get(tuple(sorted((id_a, v))), np.nan)
                for v in vecinos_ids
            ]
            avg_dist = np.nanmean(dists) if dists else np.nan
            std_dist = np.nanstd(dists)  if len(dists) > 1 else 0.0

            # constantes de fuerza de bonds
            fcs_bond = [
                fc_bonds.get(tuple(sorted((id_a, v))), np.nan)
                for v in vecinos_ids
            ]
            avg_fc_bond = np.nanmean(fcs_bond) if fcs_bond else np.nan

            # ángulos donde el átomo es el central (aj)
            angulos_atomo = [
                ang for ang in mol["angles"] if ang["aj"] == id_a
            ]
            avg_angle  = np.nanmean([a["angle"]    for a in angulos_atomo]) if angulos_atomo else np.nan
            avg_fc_ang = np.nanmean([a["fc_angle"] for a in angulos_atomo]) if angulos_atomo else np.nan

            registros.append(
                {
                    "molecule":              mol["molecule_name"],
                    "element":               atom["elemento"],
                    "mass":                  atom["masa"],
                    "coordination":          len(vecinos_ids),
                    "neighbor_config":       "-".join(vecinos_elem),
                    "n_dihedrals":           dih_count[id_a],
                    "is_planar":             1 if id_a in mol["impropers"] else 0,
                    "avg_bond_len_A":        round(avg_dist, 4)   if not np.isnan(avg_dist) else np.nan,
                    "std_bond_len_A":        round(std_dist, 4),
                    "avg_force_const_bond":  round(avg_fc_bond, 2) if not np.isnan(avg_fc_bond) else np.nan,
                    "avg_angle_deg":         round(avg_angle, 2)   if not np.isnan(avg_angle) else np.nan,
                    "avg_force_const_angle": round(avg_fc_ang, 2)  if not np.isnan(avg_fc_ang) else np.nan,
                    "target_atom_type":      atom["atom_type"],
                    "target_charge":         atom["carga"],
                }
            )

    return pd.DataFrame(registros)


# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent
    BASE       = SCRIPT_DIR.parent
    ruta_itp   = BASE / "input" / ".itp"

    print(f"📂 Buscando .itp en: {ruta_itp}")

    df = crear_base_datos(ruta_itp)

    output = SCRIPT_DIR / "base_final_v2.csv"
    df.to_csv(output, index=False)

    print(f"✅ Base generada con {len(df)} ejemplos.")
    print(f"📄 Guardado en: {output}")
    print(f"\n📋 Columnas generadas: {list(df.columns)}")
    print(f"\n🔍 Primeras filas:")
    print(df.head())
