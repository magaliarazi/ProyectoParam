import pandas as pd
import numpy as np

# =========================
# 1. Cargar datasets
# =========================
df_tutora = pd.read_csv("base_datos_flor_con_source.csv")
df_online = pd.read_csv("base_datos_nomodif_con_source.csv")

print("=== Tamaño de los datasets ===")
print(f"Tutora : {len(df_tutora)} átomos")
print(f"Online : {len(df_online)} átomos")

# =========================
# 2. Función de resumen
# =========================
def resumen_dataset(df, nombre):
    print(f"\n==============================")
    print(f"Resumen dataset: {nombre}")
    print(f"==============================")

    print("\n-- Coordinación (proporciones) --")
    print(df['coordination'].value_counts(normalize=True).sort_index())

    print("\n-- avg_bond_len --")
    print(df['avg_bond_len'].describe())
    print("NaN:", df['avg_bond_len'].isna().sum())

    print("\n-- avg_angle --")
    print(df['avg_angle'].describe())
    print("NaN:", df['avg_angle'].isna().sum())

    print("\n-- is_planar --")
    print(df['is_planar'].value_counts(normalize=True))

    print("\n-- Elementos más comunes --")
    print(df['element'].value_counts().head(10))


# =========================
# 3. Ejecutar resumen
# =========================
resumen_dataset(df_tutora, "Tutora")
resumen_dataset(df_online, "Online")

# =========================
# 4. Comparación directa
# =========================
print("\n==============================")
print("Comparación directa (medias)")
print("==============================")

comparacion = pd.DataFrame({
    "tutora_mean": [
        df_tutora['avg_bond_len'].mean(),
        df_tutora['avg_angle'].mean(),
        df_tutora['coordination'].mean()
    ],
    "online_mean": [
        df_online['avg_bond_len'].mean(),
        df_online['avg_angle'].mean(),
        df_online['coordination'].mean()
    ]
}, index=["avg_bond_len", "avg_angle", "coordination"])

print(comparacion)

print("\nDiferencia relativa (%):")
print(100 * (comparacion['online_mean'] - comparacion['tutora_mean']) / comparacion['tutora_mean'])
