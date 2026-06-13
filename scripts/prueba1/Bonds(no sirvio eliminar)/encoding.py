import pandas as pd
import json
from pathlib import Path

# ----------------------------
# PATHS
# ----------------------------
CSV_PATH = "dataset_bonds2.csv"
MAPS_PATH = Path("/home/marazi/proyectoParam/scripts/prueba1/Preprocesamiento/maps_path1/label_maps.json")

# ----------------------------
# CARGAR DATASET
# ----------------------------
df = pd.read_csv(CSV_PATH)

# ----------------------------
# CARGAR DICCIONARIOS
# ----------------------------
with open(MAPS_PATH, "r") as f:
    label_maps = json.load(f)

# ----------------------------
# APLICAR MAPPINGS
# ----------------------------
df["type_i"] = df["type_i"].map(label_maps["target_atom_type"])
df["type_j"] = df["type_j"].map(label_maps["target_atom_type"])
df["element_i"] = df["element_i"].map(label_maps["element"])
df["element_j"] = df["element_j"].map(label_maps["element"])
# ----------------------------
# CHECK DE ERRORES
# ----------------------------
print("NaNs por columna:")
print(df.isna().sum())

# opcional: reemplazar NaNs
df = df.fillna(-1)

# ----------------------------
# GUARDAR
# ----------------------------
output_path = "bond_dataset_encoded2.csv"
df.to_csv(output_path, index=False)

print(f"✔ Dataset codificado guardado en {output_path}")