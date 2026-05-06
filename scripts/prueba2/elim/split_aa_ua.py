"""
split_aa_ua.py
--------------
Divide el dataset.csv en dos tablas:
  - All-Atom (AA):     C desnudos + H no polares (HC) + todo lo demás
  - United-Atom (UA):  C/CH/CH2/CH3 + H polares (HS14, unido a N u O) + todo lo demás

Uso:
    python split_aa_ua.py --input dataset.csv --output_aa aa.csv --output_ua ua.csv

Correr ANTES del preprocesamiento, sobre el dataset.csv crudo.
"""

import argparse
import pandas as pd

# ── Definición de tipos por modelo ───────────────────────────────────────────

# Tipos de C que van a All-Atom (carbono sin hidrógenos implícitos)
AA_C_TYPES = {"C"}

# Tipos de H que van a All-Atom (hidrógenos no polares, unidos a C)
AA_H_TYPES = {"HC"}

# Tipos de C que van a United-Atom (carbonos con H implícitos + C desnudo)
UA_C_TYPES = {"C", "CH", "CH2", "CH3"}

# Tipos de H que van a United-Atom (solo polares: unidos a N u O)
# Agregá acá cualquier tipo de H polar que aparezca en tus .itp
UA_H_TYPES = {"HS14", "H", "HS"}


def dividir(input_csv: str, output_aa: str, output_ua: str):
    df = pd.read_csv(input_csv)
    df.columns = df.columns.str.strip()

    # Verificar columnas necesarias
    for col in ["element", "atomtype"]:
        if col not in df.columns:
            raise ValueError(f"Columna '{col}' no encontrada en el dataset.")

    # ── Máscaras base ─────────────────────────────────────────────────────────
    es_C    = df["element"] == "C"
    es_H    = df["element"] == "H"
    es_otro = ~es_C & ~es_H

    # ── ALL-ATOM ──────────────────────────────────────────────────────────────
    mask_aa = (
        es_otro |
        (es_C & df["atomtype"].isin(AA_C_TYPES)) |
        (es_H & df["atomtype"].isin(AA_H_TYPES))
    )
    table_aa = df[mask_aa].copy()

    # ── UNITED-ATOM ───────────────────────────────────────────────────────────
    mask_ua = (
        es_otro |
        (es_C & df["atomtype"].isin(UA_C_TYPES)) |
        (es_H & df["atomtype"].isin(UA_H_TYPES))
    )
    table_ua = df[mask_ua].copy()

    # ── Guardar ───────────────────────────────────────────────────────────────
    table_aa.to_csv(output_aa, index=False)
    table_ua.to_csv(output_ua, index=False)

    # ── Resumen ───────────────────────────────────────────────────────────────
    total = len(df)

    print("=" * 55)
    print("  TABLA ALL-ATOM (AA)")
    print("=" * 55)
    print(f"  Filas totales : {len(table_aa)} / {total}")
    print(f"  C desnudos    : {(es_C & df['atomtype'].isin(AA_C_TYPES)).sum()}")
    print(f"  HC (H no pol) : {(es_H & df['atomtype'].isin(AA_H_TYPES)).sum()}")
    print(f"  Otros (N,O..) : {es_otro.sum()}")
    print()
    print(f"  Distribución de atomtype:")
    print(table_aa["atomtype"].value_counts().to_string())

    print()
    print("=" * 55)
    print("  TABLA UNITED-ATOM (UA)")
    print("=" * 55)
    print(f"  Filas totales : {len(table_ua)} / {total}")
    print(f"  C/CH/CH2/CH3  : {(es_C & df['atomtype'].isin(UA_C_TYPES)).sum()}")
    print(f"  H polar       : {(es_H & df['atomtype'].isin(UA_H_TYPES)).sum()}")
    print(f"  Otros (N,O..) : {es_otro.sum()}")
    print()
    print(f"  Distribución de atomtype:")
    print(table_ua["atomtype"].value_counts().to_string())

    print()
    print("Archivos guardados:")
    print(f"  → {output_aa}")
    print(f"  → {output_ua}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Divide el dataset en tablas All-Atom y United-Atom"
    )
    parser.add_argument("--input",     "-i", required=True,               help="CSV de entrada (dataset.csv crudo)")
    parser.add_argument("--output_aa", "-a", default="dataset_aa.csv",    help="Salida All-Atom")
    parser.add_argument("--output_ua", "-u", default="dataset_ua.csv",    help="Salida United-Atom")
    args = parser.parse_args()

    dividir(args.input, args.output_aa, args.output_ua)