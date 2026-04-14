import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 📥 1. Cargar modelo, scaler y features
MODELO_NOMBRE = "UA"   # o AA_charge
data = joblib.load(f"/home/marazi/proyectoParam/scripts/prueba1/ModeloTargetCharge/analisis_regresion_{MODELO_NOMBRE}_charge/modelo_completo.pkl") ##cambiar UA Por AA

model = data["model"]
scaler = data["scaler"]
features = data["features"]

print("✅ Modelo cargado correctamente")
OUTPUT_DIR = Path(f"validacion_regresion_{MODELO_NOMBRE}")
OUTPUT_DIR.mkdir(exist_ok=True)

# 📂 2. Cargar nuevos datos
df_new = pd.read_csv(f"/home/marazi/proyectoParam/scripts/prueba1/Preprocesamiento/validacion_{MODELO_NOMBRE}.csv") #cambiar UA por AA

print(f"📊 Datos cargados: {df_new.shape}")

# 🔍 3. Verificar que estén las columnas necesarias
missing_cols = [col for col in features if col not in df_new.columns]

if missing_cols:
    raise ValueError(f"❌ Faltan columnas en el dataset: {missing_cols}")

# 📐 4. Seleccionar features
X_new = df_new[features]

# ⚖️ 5. Escalar datos (usar SOLO transform)
X_new_scaled = scaler.transform(X_new)

# 🤖 6. Predecir
y_pred = model.predict(X_new_scaled)

# 💾 7. Guardar resultados
df_new["predicted_charge"] = y_pred

df_new.to_csv(OUTPUT_DIR / "predicciones.csv", index=False)

print("🎯 Predicciones realizadas y guardadas en 'predicciones.csv'")

# 📊 8. (Opcional) Evaluar si existe la columna real
if "target_charge" in df_new.columns:
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    y_true = df_new["target_charge"]

    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print("\n📊 Evaluación:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")

# ======================================
# 📊 REAL vs PREDICHO (DATOS EXTERNOS)
# ======================================
if "target_charge" in df_new.columns:

    errores = np.abs(y_true - y_pred)

    plt.figure(figsize=(6,6))
    scatter = plt.scatter(y_true, y_pred, c=errores)

    plt.xlabel("Valor real")
    plt.ylabel("Predicción")
    plt.title("Real vs Predicho — Datos externos")

    # Línea ideal
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    # Barra de color
    plt.colorbar(scatter, label="Error absoluto")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "real_vs_predicho_externo.png", dpi=150)
    plt.show()

    print("\nErrores individuales:")
    for e in errores:
        print(f"{e:.4f}")

    print("\nError promedio:", errores.mean())
    print("Error máximo:", errores.max())

# ======================================
# 🔬 ANÁLISIS DE ERROR POR TIPO DE ÁTOMO
# ======================================

# 👉 Cambiar esto por el nombre real de tu columna
atom_column = "target_atom_type"  # ej: "atom", "element", etc.

if atom_column in df_new.columns:

        df_new["error_abs"] = errores

        # 📊 Promedio de error por átomo
        error_por_atomo = df_new.groupby(atom_column)["error_abs"].mean().sort_values(ascending=False)

        print("\n🔬 Error promedio por tipo de átomo:")
        print(error_por_atomo)

        # 📈 Gráfico
        plt.figure(figsize=(8,5))
        error_por_atomo.plot(kind="bar")

        plt.ylabel("Error absoluto promedio")
        plt.title("Error por tipo de átomo")

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "error_por_atomo.png", dpi=150)
        plt.show()

else:
    print(f"\n⚠️ No se encontró la columna '{atom_column}' para analizar errores por átomo")