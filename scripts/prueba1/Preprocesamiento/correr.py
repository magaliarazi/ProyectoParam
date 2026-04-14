import pandas as pd

df = pd.read_csv("base_ve.csv")

mapeo = {
    "CPos":  "C",
    "OEOpt": "OE",
    "OAlc":  "OA"
}

df["target_atom_type"] = df["target_atom_type"].replace(mapeo)
df.to_csv("base_ve_mapeada.csv", index=False)
print("✅ Tipos mapeados correctamente")
print(df["target_atom_type"].value_counts().sort_index())
