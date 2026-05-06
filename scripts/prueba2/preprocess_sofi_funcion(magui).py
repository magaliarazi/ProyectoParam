"python preprocess_sofi_funcion.py -i nombre archivo de entrado -o nombre carpeta de salida"

import argparse
import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


def preprocess_dataset(input_path, output_dir):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── CONFIG ─────────────────────────────
    ID_COLS = ["molecule", "atom_id", "atom_name"]
    TARGET_CLASS = "atomtype"
    TARGET_REG   = "charge"

    ONEHOT_COLS = ["tripos_type", "element", "neighbor_config", "neighbor2_config"]
    BOOL_COLS = ["is_planar", "is_in_ring", "is_aromatic"]

    SCALABLE_NUMERIC_COLS = [
        "mass", "avg_bond_len_A", "avg_angle_deg", "n_dihedrals",
        "gasteiger_charge", "formal_charge", "degree_of_unsat"
    ]

    # ── CARGAR ────────────────────────────
    df = pd.read_csv(input_path)
    print(f"📥 Leyendo {input_path} ({len(df)} filas)")

    # ── TARGETS ───────────────────────────
    le = LabelEncoder()
    y_class = le.fit_transform(df[TARGET_CLASS].astype(str))
    y_reg = df[TARGET_REG].astype(float).values

    y_df = pd.DataFrame({
        "atomtype_encoded": y_class,
        "atomtype_label": df[TARGET_CLASS].values,
        "charge": y_reg,
    })

    # ── FEATURES ──────────────────────────
    drop_cols = ID_COLS + [TARGET_CLASS, TARGET_REG]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()

    # Boolean → int
    for col in BOOL_COLS:
        if col in X.columns:
            X[col] = X[col].map({True: 1, False: 0, 1: 1, 0: 0}).fillna(0).astype(int)

    # One-hot
    ohe_encoders = {}
    for col in ONEHOT_COLS:
        if col in X.columns:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            X[col] = X[col].astype(str).fillna("none")

            encoded = ohe.fit_transform(X[[col]])
            col_names = [f"{col}__{cat}" for cat in ohe.categories_[0]]

            ohe_df = pd.DataFrame(encoded, columns=col_names, index=X.index)
            X = pd.concat([X, ohe_df], axis=1).drop(columns=[col])

            ohe_encoders[col] = ohe

    # Escalado
    cols_to_scale = [
        c for c in SCALABLE_NUMERIC_COLS
        if c in X.columns and X[c].nunique() > 1
    ]

    scaler = StandardScaler()
    if cols_to_scale:
        X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])

    # ── GUARDAR ───────────────────────────
    dataset_path = output_dir / "dataset_processed.csv"
    enc_path     = output_dir / "encoders.pkl"
    scaler_path  = output_dir / "scaler.pkl"

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

    print("\n✅ Listo")
    print(f"📁 Output: {output_dir}")
    print(f"📊 Filas: {len(full_df)} | Columnas: {len(full_df.columns)}")


# ── CLI ─────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocesamiento de dataset molecular")

    parser.add_argument("--input", "-i", required=True, help="Archivo CSV de entrada")
    parser.add_argument("--output", "-o", default="processed", help="Carpeta de salida")

    args = parser.parse_args()

    preprocess_dataset(args.input, args.output)