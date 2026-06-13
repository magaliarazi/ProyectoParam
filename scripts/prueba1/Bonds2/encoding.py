import pandas as pd
import json
from pathlib import Path

# ----------------------------
# PATHS
# ----------------------------
CSV_PATH = "bonds_dataset.csv"
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

print("Valores únicos en element_j no encontrados en encoder:")
print(df[~df['element_j'].isin(label_maps['element'].keys())]['element_j'].unique())

# ----------------------------
# APLICAR MAPPINGS
# ----------------------------

df['molecule']  = df['molecule'].map(label_maps['molecule'])
df['element_i'] = df['element_i'].map(label_maps['element'])
df['element_j'] = df['element_j'].map(label_maps['element'])
df['type_i']    = df['type_i'].map(label_maps['target_atom_type'])
df['type_j']    = df['type_j'].map(label_maps['target_atom_type'])
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
