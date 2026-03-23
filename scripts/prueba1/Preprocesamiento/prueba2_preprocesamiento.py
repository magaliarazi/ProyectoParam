import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# CONFIGURACIÓN DE RUTAS
# Ahora definimos una lista con los archivos que queremos procesar
INPUT_FILES = ["tabla_all_atom.csv", "tabla_united_atom.csv"]
OUTPUT_DIR = Path("output2")
MAPS_DIR = Path("maps_path2")

# ==============================
# FUNCIONES DE PROCESAMIENTO (Se mantienen igual)
# ==============================
def clean_numeric_columns(df):
    num_cols = ["mass", "coordination", "avg_bond_len", "avg_angle", "target_charge", "is_planar"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "."), errors="coerce")
    return df

def encode_categoricals(df):
    cat_cols = ["molecule", "element", "neighbor_config", "target_atom_type"]
    label_maps = {}
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_maps[col] = {str(cls): int(code) for cls, code in zip(le.classes_, le.transform(le.classes_))}
    return df, label_maps

def create_neighbor_features(df, original_df):
    if "neighbor_config" in original_df.columns:
        df["C_count"] = original_df["neighbor_config"].astype(str).str.count("C")
        df["H_count"] = original_df["neighbor_config"].astype(str).str.count("H")
        df["O_count"] = original_df["neighbor_config"].astype(str).str.count("O")
        df["N_count"] = original_df["neighbor_config"].astype(str).str.count("N")
    return df

def binarize_is_planar(df):
    if "is_planar" in df.columns:
        df["is_planar"] = (df["is_planar"] > 0).astype(int)
    return df

def filter_rare_classes(df):
    counts = df["target_atom_type"].value_counts()
    valid_classes = counts[counts >= 2].index
    return df[df["target_atom_type"].isin(valid_classes)].copy()

# ==============================
# LÓGICA DE PROCESAMIENTO POR ARCHIVO
# ==============================
def process_file(file_path):
    print(f"\n🚀 Procesando: {file_path}")
    
    if not Path(file_path).exists():
        print(f"⚠️ El archivo {file_path} no existe. Saltando...")
        return

    df = pd.read_csv(file_path)
    df_original = df.copy()

    df = df.fillna("unknown")
    df = clean_numeric_columns(df)
    df, label_maps = encode_categoricals(df)
    df = create_neighbor_features(df, df_original)
    df = binarize_is_planar(df)
    
    df_numeric = df.select_dtypes(include=[np.number])
    df_filtered = filter_rare_classes(df_numeric)

    # Preparar X para el escalador
    X = df_filtered.drop(columns=["target_atom_type", "molecule"], errors="ignore")
    scaler = StandardScaler()
    scaler.fit_transform(X)

    # Definir nombres de salida basados en el archivo original
    # Ejemplo: tabla_all_atom -> preprocessed_all_atom.csv
    suffix = Path(file_path).stem.replace("tabla_", "")
    
    OUTPUT_CSV = OUTPUT_DIR / f"preprocessed_{suffix}.csv"
    MAPS_JSON = MAPS_DIR / f"label_maps_{suffix}.json"
    SCALER_BIN = MAPS_DIR / f"scaler_{suffix}.joblib"

    # Crear carpetas
    OUTPUT_DIR.mkdir(exist_ok=True)
    MAPS_DIR.mkdir(exist_ok=True)

    # Guardar
    df_filtered.to_csv(OUTPUT_CSV, index=False)
    with open(MAPS_JSON, "w") as f:
        json.dump(label_maps, f, indent=2)
    joblib.dump(scaler, SCALER_BIN)

    print(f"✅ Completado. Archivos guardados con sufijo: {suffix}")

def main():
    for file in INPUT_FILES:
        process_file(file)
    print("\n✨ Proceso finalizado para todas las tablas.")

if __name__ == "__main__":
    main()
