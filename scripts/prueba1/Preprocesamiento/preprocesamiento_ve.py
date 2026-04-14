import pandas as pd
import numpy as np
import json
from pathlib import Path
import joblib

# ==============================
# RUTAS
# ==============================
INPUT_CSV   = "/home/marazi/proyectoParam/scripts/prueba1/Preprocesamiento/base_ve_mapeada.csv"  # CSV de validación externa
OUTPUT_CSV  = "base_ve_preprocesada.csv"       # CSV preprocesado listo para predecir
MAPS_PATH   = "maps_path1/label_maps.json"     # diccionario ya existente del training
SCALER_PATH = "maps_path1/scaler.joblib"       # scaler ya entrenado del training


# ==============================
# LIMPIEZA NUMÉRICA
# ==============================
def clean_numeric_columns(df):
    num_cols = [
        "mass",
        "coordination",
        "avg_bond_len_A",
        "avg_angle_deg",
        "target_charge",
        "is_planar",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "."),
                errors="coerce",
            )
    return df


# ==============================
# ENCODING CON DICCIONARIO FIJO
# ==============================
def encode_with_fixed_maps(df, fixed_maps):
    for col, mapping in fixed_maps.items():
        if col in df.columns:
            df[col] = df[col].astype(str).map(mapping)
            n_nan = df[col].isna().sum()
            if n_nan > 0:
                print(f"  ⚠️  {col}: {n_nan} valores no encontrados en el diccionario")
            else:
                print(f"  ✅ {col}: OK")
    return df


# ==============================
# FEATURES DE NEIGHBOR
# ==============================
def create_neighbor_features(df, original_df):
    if "neighbor_config" in original_df.columns:
        df["C_count"] = original_df["neighbor_config"].astype(str).str.count("C")
        df["H_count"] = original_df["neighbor_config"].astype(str).str.count("H")
        df["O_count"] = original_df["neighbor_config"].astype(str).str.count("O")
        df["N_count"] = original_df["neighbor_config"].astype(str).str.count("N")
        print("  🧪 Features de neighbor creadas")
    return df


# ==============================
# BINARIZAR PLANARIDAD
# ==============================
def binarize_is_planar(df):
    if "is_planar" in df.columns:
        df["is_planar"] = (df["is_planar"] > 0).astype(int)
    return df


# ==============================
# MAIN
# ==============================
def main():
    print("📥 Cargando datos de validación externa...")
    df = pd.read_csv(INPUT_CSV)
    df.columns = df.columns.str.strip()
    df_original = df.copy()

    print(f"   Filas: {len(df)} | Columnas: {list(df.columns)}")

    # 1️⃣ Cargar diccionario y scaler ya existentes del training
    print("\n🗺️  Cargando diccionario y scaler del training...")
    with open(MAPS_PATH, "r") as f:
        fixed_maps = json.load(f)
    scaler = joblib.load(SCALER_PATH)
    print("   ✅ Diccionario y scaler cargados")

    # 2️⃣ Fill NaN
    df = df.fillna("unknown")

    # 3️⃣ Numéricos
    df = clean_numeric_columns(df)

    # 4️⃣ Encoding con diccionario fijo (NO re-entrenar)
    print("\n🔤 Codificando columnas con diccionario del training...")
    df = encode_with_fixed_maps(df, fixed_maps)

    # 5️⃣ Features de neighbor
    df = create_neighbor_features(df, df_original)

    # 6️⃣ is_planar binario
    df = binarize_is_planar(df)

    # 7️⃣ Solo numéricos
    df_numeric = df.select_dtypes(include=[np.number])

    # 8️⃣ Separar X e y
    y = df_numeric["target_atom_type"]
    X = df_numeric.drop(columns=["target_atom_type", "molecule", "avg_angle_deg"], errors="ignore")

    print(f"\n✅ Features listas: {list(X.columns)}")
    print(f"   Shape X: {X.shape}")

    # 9️⃣ Escalar con scaler del training (NO re-entrenar)
    print("\n🔧 Escalando con scaler del training...")
    X_scaled = scaler.transform(X)

    # Chequeos
    print("\n🔍 Chequeos de calidad:")
    print(f"   NaNs en X: {np.isnan(X_scaled).sum()}")
    print(f"   NaNs en y: {y.isna().sum()}")

    # Guardar
    df_numeric.to_csv(OUTPUT_CSV, index=False)
    print(f"\n💾 Datos preprocesados guardados en: {OUTPUT_CSV}")
    print("\n✅ Preprocesamiento de validación externa completado")


if __name__ == "__main__":
    main()