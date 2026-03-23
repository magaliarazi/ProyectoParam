import pandas as pd

# ── Cargar datos ──────────────────────────────────────────────────────────────
df = pd.read_csv("/home/marazi/proyectoParam/scripts/prueba1/Preprocesamiento/output1/preprocessed_data.csv")   # <-- cambiá el nombre si es necesario
df.columns = df.columns.str.strip()

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

# ── Máscaras base ─────────────────────────────────────────────────────────────
es_C   = df["element"] == ELEM_C
es_H   = df["element"] == ELEM_H
es_otro = ~es_C & ~es_H   # N, O, F, S, Cl → igual en ambas tablas

# ── TABLA ALL-ATOM (AA) ───────────────────────────────────────────────────────
# - Todo lo que no es C ni H
# - C desnudo (target_atom_type == 0)
# - H no polar, HC (target_atom_type == 9)

mask_aa = (
    es_otro |
    (es_C & (df["target_atom_type"] == TAT_C)) |
    (es_H  & (df["target_atom_type"] == TAT_HC))
)
table_aa = df[mask_aa].copy()

# ── TABLA UNITED-ATOM (UA) ────────────────────────────────────────────────────
# - Todo lo que no es C ni H
# - C, CH, CH2, CH3 (target_atom_type en {0, 2, 3, 4})
# - Solo H polar (target_atom_type en {8, 10})

mask_ua = (
    es_otro |
    (es_C & (df["target_atom_type"].isin([TAT_C, TAT_CH, TAT_CH2, TAT_CH3]))) |
    (es_H  & (df["target_atom_type"].isin([TAT_H, TAT_HS])))
)
table_ua = df[mask_ua].copy()

# ── Guardar ───────────────────────────────────────────────────────────────────
table_aa.to_csv("tabla_all_atom.csv",    index=False)
table_ua.to_csv("tabla_united_atom.csv", index=False)

# ── Resumen ───────────────────────────────────────────────────────────────────
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
print("  → tabla_all_atom.csv")
print("  → tabla_united_atom.csv")