import pandas as pd
import argparse

# ── Códigos del diccionario ───────────────────────────────────────────────────
ELEM_C  = 0
ELEM_H  = 3

TAT_C   = 0    # C desnudo
TAT_CH  = 2    # CH
TAT_CH2 = 3    # CH2
TAT_CH3 = 4    # CH3
TAT_HC  = 9    # H no polar (unido a C)
TAT_H   = 8    # H polar (unido a N u O)
TAT_HS  = 10   # H polar (unido a S)


def dividir(input_csv, output_aa, output_ua):
    # ── Cargar datos ──────────────────────────────────────────────────────────
    df = pd.read_csv(input_csv)
    df.columns = df.columns.str.strip()

    # ── Máscaras base ─────────────────────────────────────────────────────────
    es_C    = df["element"] == ELEM_C
    es_H    = df["element"] == ELEM_H
    es_otro = ~es_C & ~es_H

    # ── TABLA ALL-ATOM (AA) ───────────────────────────────────────────────────
    mask_aa = (
        es_otro |
        (es_C & (df["target_atom_type"] == TAT_C)) |
        (es_H  & (df["target_atom_type"] == TAT_HC))
    )
    table_aa = df[mask_aa].copy()

    # ── TABLA UNITED-ATOM (UA) ────────────────────────────────────────────────
    mask_ua = (
        es_otro |
        (es_C & (df["target_atom_type"].isin([TAT_C, TAT_CH, TAT_CH2, TAT_CH3]))) |
        (es_H  & (df["target_atom_type"].isin([TAT_H, TAT_HS])))
    )
    table_ua = df[mask_ua].copy()

    # ── Guardar ───────────────────────────────────────────────────────────────
    table_aa.to_csv(output_aa, index=False)
    table_ua.to_csv(output_ua, index=False)

    # ── Resumen ───────────────────────────────────────────────────────────────
    print("=" * 55)
    print("  TABLA ALL-ATOM (AA)")
    print("=" * 55)
    print(f"  Filas totales : {len(table_aa)}")
    print(f"  C desnudos    : {(es_C & (df['target_atom_type'] == TAT_C)).sum()}")
    print(f"  HC (H no pol) : {(es_H & (df['target_atom_type'] == TAT_HC)).sum()}")
    print(f"  Otros (N,O..) : {es_otro.sum()}")

    print()
    print("=" * 55)
    print("  TABLA UNITED-ATOM (UA)")
    print("=" * 55)
    print(f"  Filas totales : {len(table_ua)}")
    print(f"  C/CH/CH2/CH3  : {(es_C & df['target_atom_type'].isin([TAT_C,TAT_CH,TAT_CH2,TAT_CH3])).sum()}")
    print(f"  H polar       : {(es_H & df['target_atom_type'].isin([TAT_H,TAT_HS])).sum()}")
    print(f"  Otros (N,O..) : {es_otro.sum()}")

    print()
    print("Archivos guardados:")
    print(f"  → {output_aa}")
    print(f"  → {output_ua}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Divide un CSV preprocesado en tablas All-Atom y United-Atom"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Ruta al CSV de entrada (ej: preprocessed_data.csv)"
    )
    parser.add_argument(
        "--output_aa",
        default="tabla_all_atom.csv",
        help="Ruta de salida para la tabla All-Atom (default: tabla_all_atom.csv)"
    )
    parser.add_argument(
        "--output_ua",
        default="tabla_united_atom.csv",
        help="Ruta de salida para la tabla United-Atom (default: tabla_united_atom.csv)"
    )

    args = parser.parse_args()
    dividir(args.input, args.output_aa, args.output_ua)