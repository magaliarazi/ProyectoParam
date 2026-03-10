import pandas as pd

df_tutora = pd.read_csv("base_datos_flor.csv")
df_online = pd.read_csv("base_datos_nomodif.csv")  

df_tutora['source'] = 'tutora'
df_online['source'] = 'online'

df_tutora.to_csv("base_datos_flor_con_source.csv", index=False)
df_online.to_csv("base_datos_nomodif_con_source.csv", index=False)

