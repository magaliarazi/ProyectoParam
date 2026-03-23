import pandas as pd

df = pd.read_csv("/home/marazi/proyectoParam/scripts/prueba1/Preprocesamiento/output1/preprocessed_data.csv")
df.columns = df.columns.str.strip()

print(df["target_atom_type"].value_counts().sort_index())