import argparse
import pickle
import sys
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# 1. IDENTIFICADORES (Se eliminan)
ID_COLS = ["molecule", "atom_id", "atom_name"]

# 2. TARGETS
TARGET_CLASS = "atomtype"   # Clasificación
TARGET_REG   = "charge"     # Regresión

# 3. CATEGÓRICAS -> One-Hot (Sin escalar después)
ONEHOT_COLS = ["tripos_type", "element", "neighbor_config", "neighbor2_config"]

# 4. BOOLEANAS -> 0/1 (Sin escalar después)
BOOL_COLS = ["is_planar", "is_in_ring", "is_aromatic"]

# 5. NUMÉRICAS CONTINUAS -> StandardScaler
SCALABLE_NUMERIC_COLS = [
    "mass", "avg_bond_len_A", "avg_angle_deg", "n_dihedrals",
    "gasteiger_charge", "formal_charge", "degree_of_unsat"
]

def main():
    parser = argparse.ArgumentParser(description="Preprocesa dataset.csv con sufijo sofi")
    parser.add_argument("--input", "-i", required=True, help="Archivo dataset.csv")
    
    # --- CAMBIO AQUÍ: Carpeta predeterminada actualizada ---
    parser.add_argument("--output_dir", "-o", default="processed_sofi", help="Carpeta de salida")
    
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"Error: '{input_path}' no existe.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    print(f"Leyendo {input_path} ({len(df)} filas)")

    # ── PROCESAR TARGETS ──────────────────────────────────────────────────
    le = LabelEncoder()
    y_class = le.fit_transform(df[TARGET_CLASS].astype(str))
    y_reg = df[TARGET_REG].astype(float).values

    y_df = pd.DataFrame({
        "atomtype_encoded": y_class,
        "atomtype_label":   df[TARGET_CLASS].values,
        "charge":           y_reg,
    })

    # ── PREPARAR FEATURES (X) ─────────────────────────────────────────────
    drop_cols = ID_COLS + [TARGET_CLASS, TARGET_REG]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()

    # 1. Tratar Booleanas (0 / 1 puro)
    for col in BOOL_COLS:
        if col in X.columns:
            X[col] = X[col].map({True: 1, False: 0, 1: 1, 0: 0}).fillna(0).astype(int)

    # 2. One-Hot Encoding
    ohe_encoders = {}
    for col in ONEHOT_COLS:
        if col in X.columns:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            X[col] = X[col].astype(str).fillna("none")
            encoded_val = ohe.fit_transform(X[[col]])
            
            col_names = [f"{col}__{cat}" for cat in ohe.categories_[0]]
            ohe_df = pd.DataFrame(encoded_val, columns=col_names, index=X.index)
            
            X = pd.concat([X, ohe_df], axis=1).drop(columns=[col])
            ohe_encoders[col] = ohe

    # 3. Escalar solo Numéricas Continuas
    cols_to_scale = [c for c in SCALABLE_NUMERIC_COLS if c in X.columns and X[c].nunique() > 1]
    scaler = StandardScaler()
    if cols_to_scale:
        X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])

    # ── GUARDAR RESULTADOS CON SUFIJO _sofi ───────────────────────────────
    dataset_path = output_dir / "dataset_processed_sofi.csv"
    enc_path     = output_dir / "encoders_sofi.pkl"
    scaler_path  = output_dir / "scaler_sofi.pkl"

    full_df = pd.concat([X, y_df], axis=1)
    full_df.to_csv(dataset_path, index=False)

    metadata = {
        "label_encoder_atomtype": le,
        "onehot_encoders": ohe_encoders,
        "scaled_cols": cols_to_scale,
        "feature_cols": X.columns.tolist()
    }

    with open(enc_path, "wb") as f:
        pickle.dump(metadata, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"\n✓ Proceso completado exitosamente.")
    print(f"✓ Los archivos están en la carpeta: {output_dir}")
    print(f"  - dataset_processed_sofi.csv")
    print(f"  - encoders_sofi.pkl")
    print(f"  - scaler_sofi.pkl")

if __name__ == "__main__":
    main()
