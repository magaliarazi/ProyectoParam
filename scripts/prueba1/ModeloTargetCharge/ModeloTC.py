import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from sklearn.model_selection import cross_val_score

# ======================================
# CONFIG
# ======================================
DATA_PATH = "/home/marazi/proyectoParam/scripts/prueba1/Preprocesamiento/tabla_united_atom_v2.csv" # cambiar all por united
MODELO_NOMBRE = "UA_charge_v2"   # o AA_charge

TEST_SIZE = 0.2
RANDOM_STATE = 42
OUTPUT_DIR = Path(f"analisis_regresion_{MODELO_NOMBRE}")

# ======================================
# MAIN
# ======================================
def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("📥 Cargando dataset...")
    df = pd.read_csv(DATA_PATH)

    # Features y target
    X = df.drop(columns=["target_charge", "target_atom_type", "molecule", 
                      "N_count", "coordination", "n_dihedrals",
                      "avg_angle_deg", "avg_force_const_angle",
                      "avg_force_const_bond", "std_bond_len_A"], errors="ignore")
    y = df["target_charge"]
    features = X.columns.tolist()

    print("\n🔁 Cross-validation (5 folds)...")

    model_cv = RandomForestRegressor(
    n_estimators=200,
    random_state=RANDOM_STATE
    )

    scores = cross_val_score(model_cv, X, y, cv=5, scoring="r2")

    print("R² por fold:", scores)
    print("R² promedio:", scores.mean())

    print(f"Shape X: {X.shape}")

    # Train / Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Escalado
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # ======================================
    # MODELO
    # ======================================

    print("\n🌳 Entrenando RandomForestRegressor...")

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=RANDOM_STATE
    )
    model.fit(X_train_scaled, y_train)

    # Predicción
    y_pred = model.predict(X_test_scaled)

    # ======================================
    # MÉTRICAS
    # ======================================
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    print("\n📊 Resultados:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")

    # ======================================
    # 1. REAL vs PREDICHO
    # ======================================
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.xlabel("Valor real")
    plt.ylabel("Predicción")
    plt.title(f"Real vs Predicho — {MODELO_NOMBRE}")

    # línea ideal
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "real_vs_predicho.png", dpi=150)
    plt.close()

    # ======================================
    # 2. DISTRIBUCIÓN DE ERRORES
    # ======================================
    errores = y_test - y_pred

    plt.figure(figsize=(6,4))
    sns.histplot(errores, kde=True)
    plt.xlabel("Error (real - predicho)")
    plt.title(f"Distribución de errores — {MODELO_NOMBRE}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "errores.png", dpi=150)
    plt.close()

    # ======================================
    # FEATURE IMPORTANCE
    # ======================================
    importances = model.feature_importances_
    feature_names = X.columns

    # Ordenar de mayor a menor
    indices = np.argsort(importances)[::-1]

    print("\n🌟 Feature importance:")
    for i in indices:
        print(f"{feature_names[i]}: {importances[i]:.4f}")

    # Gráfico
    plt.figure(figsize=(8,5))
    plt.bar(range(len(importances)), importances[indices], tick_label=feature_names[indices])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Importancia")
    plt.title("Importancia de features — Random Forest")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150)
    plt.close()
    # ======================================
    # 3. CURVA DE LOSS
    # ======================================
    #plt.figure(figsize=(6,4))
    #plt.plot(model.loss_curve_)
    #plt.xlabel("Iteraciones")
    #plt.ylabel("Loss")
    #plt.title(f"Curva de entrenamiento — {MODELO_NOMBRE}")
    #plt.tight_layout()
    #plt.savefig(OUTPUT_DIR / "loss_curve.png", dpi=150)
    #plt.close()
    joblib.dump({
        "model": model,
        "scaler": scaler,
        "features": features
    },OUTPUT_DIR / "modelo_completo.pkl")

    print(f"\n📁 Resultados guardados en: {OUTPUT_DIR}")

# ======================================
if __name__ == "__main__":
    main()