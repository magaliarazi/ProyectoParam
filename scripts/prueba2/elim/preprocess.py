"""
preprocess.py
-------------
Toma el dataset.csv generado por extract_dataset.py y produce:
  - dataset_processed.csv         → features numéricas escaladas, listas para la red neuronal y targets (atomtype como entero, charge como float)
  - encoders.pkl  → LabelEncoders y OneHotEncoders para invertir predicciones
  - scaler.pkl    → StandardScaler para invertir la escala si hace falta

Uso:
    python preprocess.py --input dataset.csv --output_dir processed/

Dependencias:
    pip install pandas scikit-learn
"""

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


# Columnas que se descartan (identificadores, no son features)
ID_COLS = ["molecule", "atom_id", "atom_name"]

# Targets
TARGET_CLASS = "atomtype"   # clasificación → LabelEncoder
TARGET_REG   = "charge"     # regresión     → float, sin transformar

# Columnas categóricas → One-Hot
ONEHOT_COLS = ["tripos_type", "element"]

# Columnas categóricas → Hash
HASH_COLS = ["neighbor_config", "neighbor2_config"]

# Columnas booleanas → 0/1
BOOL_COLS = ["is_planar", "is_in_ring", "is_aromatic"]

# Columnas numéricas continuas → StandardScaler
# (el resto que no sea ID, categórica, bool ni target)
NUMERIC_COLS = [
    "mass", "coordination", "n_dihedrals",
    "avg_bond_len_A", "avg_angle_deg",
    "C_count", "H_count", "O_count", "N_count",
    "n_electroneg_neighbors", "n_electroneg_neighbors2",
    "bonds_single", "bonds_double", "bonds_aromatic",
    "ring_size", "formal_charge", "degree_of_unsat", "gasteiger_charge",
]


def hash_string(s: str, n_buckets: int = 1024) -> int:
    """Hash determinístico de una string a un entero en [0, n_buckets)."""
    return hash(s) % n_buckets


def main():
    parser = argparse.ArgumentParser(description="Preprocesa dataset.csv para redes neuronales")
    parser.add_argument("--input",      "-i", required=True, help="Archivo dataset.csv")
    parser.add_argument("--output_dir", "-o", default="processed", help="Carpeta de salida")
    parser.add_argument("--hash_buckets", type=int, default=1024,
                        help="Número de buckets para el hash de neighbor_config (default: 1024)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"Error: '{input_path}' no existe.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Leyendo {input_path} ...")
    df = pd.read_csv(input_path)
    print(f"  {len(df)} filas · {len(df.columns)} columnas")

    # ── Separar targets ──────────────────────────────────────────────────────
    print("\nProcesando targets ...")

    # Target clasificación: atomtype → entero
    le = LabelEncoder()
    y_class = le.fit_transform(df[TARGET_CLASS].astype(str))
    print(f"  atomtype: {len(le.classes_)} clases únicas → {list(le.classes_[:8])}{'...' if len(le.classes_) > 8 else ''}")

    # Target regresión: charge → float directo
    y_reg = df[TARGET_REG].astype(float).values

    y_df = pd.DataFrame({
        "atomtype_encoded": y_class,
        "atomtype_label":   df[TARGET_CLASS].values,
        "charge":           y_reg,
    })

    # ── Descartar identificadores y targets del df de features ──────────────
    drop_cols = ID_COLS + [TARGET_CLASS, TARGET_REG]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()

    # ── Booleanas → 0/1 ─────────────────────────────────────────────────────
    print("\nConvirtiendo booleanas ...")
    for col in BOOL_COLS:
        if col in X.columns:
            X[col] = X[col].astype(bool).astype(int)
            print(f"  {col} → 0/1")

    # ── Hash de neighbor_config y neighbor2_config ───────────────────────────
    print(f"\nHasheando columnas de config (buckets={args.hash_buckets}) ...")
    for col in HASH_COLS:
        if col in X.columns:
            X[col] = X[col].astype(str).apply(lambda s: hash_string(s, args.hash_buckets))
            print(f"  {col} → entero [0, {args.hash_buckets})")

    # ── One-Hot Encoding ─────────────────────────────────────────────────────
    print("\nOne-Hot encoding ...")
    ohe_encoders = {}
    ohe_frames   = []

    for col in ONEHOT_COLS:
        if col not in X.columns:
            continue
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoded = ohe.fit_transform(X[[col]])
        col_names = [f"{col}__{cat}" for cat in ohe.categories_[0]]
        ohe_frames.append(pd.DataFrame(encoded, columns=col_names, index=X.index))
        ohe_encoders[col] = ohe
        print(f"  {col} → {len(col_names)} columnas ({list(ohe.categories_[0][:5])}{'...' if len(ohe.categories_[0]) > 5 else ''})")

    X = X.drop(columns=ONEHOT_COLS)
    if ohe_frames:
        X = pd.concat([X] + ohe_frames, axis=1)

    # ── StandardScaler sobre numéricas ───────────────────────────────────────
    print("\nEscalando columnas numéricas con StandardScaler ...")

    # Incluir también las columnas hasheadas en el escalado
    cols_to_scale = [c for c in NUMERIC_COLS + HASH_COLS if c in X.columns]

    scaler = StandardScaler()
    X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])
    print(f"  {len(cols_to_scale)} columnas escaladas")

    # ── Guardar ──────────────────────────────────────────────────────────────
    dataset_path = output_dir / "dataset_processed.csv"
    enc_path     = output_dir / "encoders.pkl"
    scaler_path  = output_dir / "scaler.pkl"

    # Todo junto: features + targets al final
    full_df = pd.concat([X, y_df], axis=1)
    full_df.to_csv(dataset_path, index=False)

    encoders = {
        "label_encoder_atomtype": le,
        "onehot_encoders":        ohe_encoders,
        "hash_buckets":           args.hash_buckets,
        "scaled_cols":            cols_to_scale,
    }
    with open(enc_path, "wb") as f:
        pickle.dump(encoders, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"\n✓ dataset_processed.csv → {dataset_path}  ({full_df.shape[0]} filas × {full_df.shape[1]} columnas)")
    print(f"✓ encoders.pkl          → {enc_path}")
    print(f"✓ scaler.pkl            → {scaler_path}")
    print(f"\nColumnas finales de X ({X.shape[1]}):")
    for i, c in enumerate(X.columns, 1):
        print(f"  {i:3}. {c}")


if __name__ == "__main__":
    main()