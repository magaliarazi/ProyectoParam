import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

INPUT_CSV   = "input1/base_final_v2.csv"
OUTPUT_CSV  = "output1/preprocessed_data_v2.csv"
MAPS_PATH   = "maps_path1/label_maps_v2.json"
SCALER_PATH = "maps_path1/scaler_v2.joblib"


# ==============================
# IMPUTAR NaN NUMÉRICOS
# Columnas con NaN por razones físicas
# (ej: átomos sin ángulos) → imputar con 0
# ==============================
def impute_numeric_nans(df):
    cols_imputar = [
        "avg_angle_deg",
        "avg_force_const_angle",
        "std_bond_len_A",
        "avg_force_const_bond",
    ]
    for col in cols_imputar:
        if col in df.columns:
            n_nan = df[col].isna().sum()
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            if n_nan > 0:
                print(f"  🔧 {col}: {n_nan} NaN imputados con 0")
    return df


# ==============================
# LIMPIEZA NUMÉRICA
# ==============================
def clean_numeric_columns(df):
    num_cols = [
        "mass", "coordination", "avg_bond_len_A",
        "avg_angle_deg", "target_charge", "is_planar",
        "std_bond_len_A", "avg_force_const_bond", "avg_force_const_angle",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "."),
                errors="coerce",
            )
    return df


# ==============================
# ENCODING CATEGÓRICO
# ==============================
def encode_categoricals(df):
    cat_cols = [
        "molecule", "element", "neighbor_config", "target_atom_type",
    ]
    label_maps = {}
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_maps[col] = {
                str(cls): int(code)
                for cls, code in zip(le.classes_, le.transform(le.classes_))
            }
            print(f"🔤 {col}: {len(le.classes_)} categorías")
    return df, label_maps


# ==============================
# FEATURES DE NEIGHBOR
# ==============================
def create_neighbor_features(df, original_df):
    if "neighbor_config" in original_df.columns:
        df["C_count"] = original_df["neighbor_config"].astype(str).str.count("C")
        df["H_count"] = original_df["neighbor_config"].astype(str).str.count("H")
        df["O_count"] = original_df["neighbor_config"].astype(str).str.count("O")
        df["N_count"] = original_df["neighbor_config"].astype(str).str.count("N")
        print("🧪 Features de neighbor creadas")
    return df


# ==============================
# BINARIZAR PLANARIDAD
# ==============================
def binarize_is_planar(df):
    if "is_planar" in df.columns:
        df["is_planar"] = (df["is_planar"] > 0).astype(int)
    return df


# ==============================
# FILTRAR CLASES RARAS
# ==============================
def filter_rare_classes(df):
    counts = df["target_atom_type"].value_counts()
    valid_classes = counts[counts >= 2].index
    df_filtered = df[df["target_atom_type"].isin(valid_classes)].copy()
    print(
        f"🧹 Registros filtrados: {len(df_filtered)} "
        f"(eliminados {len(df) - len(df_filtered)})"
    )
    return df_filtered


# ==============================
# MAIN
# ==============================
def main():
    print("📥 Cargando dataset...")

    df = pd.read_csv(INPUT_CSV)
    df_original = df.copy()

    # 1️⃣ Imputar NaN numéricos ANTES del fillna
    print("\n🔧 Imputando NaN numéricos...")
    df = impute_numeric_nans(df)

    # 2️⃣ Fill NaN restantes (categóricos)
    df = df.fillna("unknown")

    # 3️⃣ Numéricos
    df = clean_numeric_columns(df)

    # 4️⃣ Categóricos
    df, label_maps = encode_categoricals(df)

    # 5️⃣ Features extra
    df = create_neighbor_features(df, df_original)

    # 6️⃣ is_planar binario
    df = binarize_is_planar(df)

    # 7️⃣ Solo numéricos
    df_numeric = df.select_dtypes(include=[np.number])
    print("\n✅ Dataset numérico listo")
    print("Shape:", df_numeric.shape)
    print("Columnas:", list(df_numeric.columns))

    # 8️⃣ Filtrar clases raras
    df_filtered = filter_rare_classes(df_numeric)

    # ==============================
    # PREPARAR PARA ML
    # ==============================
    X = df_filtered.drop(columns=["target_atom_type", "molecule"], errors="ignore")
    y = df_filtered["target_atom_type"]

    print("\n🎯 Features para entrenamiento:")
    print(list(X.columns))

    # ==============================
    # ESCALADO
    # ==============================
    print("\n🔧 Escalando features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ==============================
    # CHECKS DE CALIDAD
    # ==============================
    print("\n🔍 Chequeos de calidad:")
    print("NaNs en X:", np.isnan(X_scaled).sum())
    print("NaNs en y:", y.isna().sum())
    print("Clases únicas:", y.nunique())
    print("\nDistribución de clases:")
    print(y.value_counts().head(10))

    # ==============================
    # GUARDAR OUTPUTS
    # ==============================
    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    Path(MAPS_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(SCALER_PATH).parent.mkdir(parents=True, exist_ok=True)

    df_filtered.to_csv(OUTPUT_CSV, index=False)

    with open(MAPS_PATH, "w") as f:
        json.dump(label_maps, f, indent=2)

    joblib.dump(scaler, SCALER_PATH)

    print(f"\n💾 Dataset guardado en: {OUTPUT_CSV}")
    print(f"🗺️ Mapas guardados en:  {MAPS_PATH}")
    print(f"⚖️ Scaler guardado en:  {SCALER_PATH}")
    print("\n✅ Preprocesamiento completado correctamente")


if __name__ == "__main__":
    main()