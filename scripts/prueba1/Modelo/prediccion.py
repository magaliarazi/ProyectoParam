import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)

# ======================================
# CONFIG
# ======================================
MODELO_NOMBRE = "UA"   # "AA" o "UA"

DATA_PATH   = f"/home/marazi/proyectoParam/scripts/prueba1/Preprocesamiento/validacion_{MODELO_NOMBRE}.csv"
MODEL_PATH  = f"/home/marazi/proyectoParam/scripts/prueba1/Modelo/Modelo/mlp_model_{MODELO_NOMBRE}.joblib"
SCALER_PATH = f"/home/marazi/proyectoParam/scripts/prueba1/Modelo/Modelo/scaler_{MODELO_NOMBRE}.joblib"
OUTPUT_DIR  = Path(f"validacion_externa_{MODELO_NOMBRE}")

LABEL_NAMES = {
    0: "C", 1: "CAro", 2: "CH1", 3: "CH2", 4: "CH3",
    5: "CL", 6: "CLAro", 7: "F", 8: "H", 9: "HC",
    10: "HS14", 11: "N", 12: "NOpt", 13: "NPri", 14: "NT",
    15: "O", 16: "OA", 17: "OE", 18: "OM", 19: "SDmso"
}

# ======================================
# MAIN
# ======================================
def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # ── Cargar datos ──────────────────────────────────────────────────────────
    print(f"📥 Cargando datos de validación externa ({MODELO_NOMBRE})...")
    df = pd.read_csv(DATA_PATH)

    # Verificar NaN en target_atom_type
    print("NaN en target_atom_type:", df["target_atom_type"].isna().sum())
    print("Valores únicos:", df["target_atom_type"].unique())

    # Eliminar filas con NaN en target y convertir a entero
    df = df.dropna(subset=["target_atom_type"])
    df["target_atom_type"] = df["target_atom_type"].astype(int)

    print(f"   Filas: {len(df)}")

    X = df.drop(columns=["target_atom_type", "molecule", "target_charge",
                          "avg_angle_deg"], errors="ignore")
    y = df["target_atom_type"]

    print(f"   Features: {list(X.columns)}")
    print(f"   Clases reales: {sorted(y.unique())}")
    # ── Cargar modelo y scaler ────────────────────────────────────────────────
    print(f"\n🔧 Cargando modelo y scaler...")
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("   ✅ Modelo y scaler cargados")

    # ── Escalar con scaler del training ───────────────────────────────────────
    X_scaled = scaler.transform(X)

    # ── Predecir ──────────────────────────────────────────────────────────────
    y_pred = model.predict(X_scaled)
    acc = accuracy_score(y, y_pred)

    print(f"\n🎯 Accuracy validación externa: {acc:.4f}")
    print("\n📋 Reporte de Clasificación:")
    print(classification_report(y, y_pred, zero_division=0))

    # ── Matriz de confusión ───────────────────────────────────────────────────
    clases_presentes = sorted(y.unique())
    nombres = [LABEL_NAMES.get(c, str(c)) for c in clases_presentes]

    cm = confusion_matrix(y, y_pred, labels=clases_presentes)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=nombres, yticklabels=nombres,
                linewidths=0.5, ax=ax)
    ax.set_xlabel("Predicho", fontsize=12)
    ax.set_ylabel("Real", fontsize=12)
    ax.set_title(f"Matriz de Confusión — Validación Externa {MODELO_NOMBRE}", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"confusion_matrix_ve_{MODELO_NOMBRE}.png", dpi=150)
    plt.close()
    print(f"\n📊 Confusion matrix → {OUTPUT_DIR}/confusion_matrix_ve_{MODELO_NOMBRE}.png")

    # ── F1 por clase ──────────────────────────────────────────────────────────
    f1_scores = f1_score(y, y_pred, labels=clases_presentes,
                         average=None, zero_division=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    colores = ["tomato" if f < 0.9 else "steelblue" for f in f1_scores]
    bars = ax.bar(nombres, f1_scores, color=colores, edgecolor="white")
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.axhline(y=0.9, color="gray", linestyle="--",
               linewidth=1, label="Umbral 0.90")
    ax.set_xlabel("Tipo de átomo", fontsize=12)
    ax.set_ylabel("F1-score", fontsize=12)
    ax.set_title(f"F1-score por clase — Validación Externa {MODELO_NOMBRE}", fontsize=14)
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"f1_por_clase_ve_{MODELO_NOMBRE}.png", dpi=150)
    plt.close()
    print(f"📊 F1 por clase → {OUTPUT_DIR}/f1_por_clase_ve_{MODELO_NOMBRE}.png")

    # ── Detalle de predicciones ───────────────────────────────────────────────
    resultado = X.copy()
    resultado["real"]            = y.values
    resultado["predicho"]        = y_pred
    resultado["nombre_real"]     = resultado["real"].map(LABEL_NAMES)
    resultado["nombre_predicho"] = resultado["predicho"].map(LABEL_NAMES)
    resultado["correcto"]        = y.values == y_pred

    resultado.to_csv(OUTPUT_DIR / f"predicciones_ve_{MODELO_NOMBRE}.csv", index=False)
    print(f"💾 Predicciones → {OUTPUT_DIR}/predicciones_ve_{MODELO_NOMBRE}.csv")

    # ── Resumen de errores ────────────────────────────────────────────────────
    errores = resultado[~resultado["correcto"]]
    print(f"\n❌ Errores: {len(errores)} de {len(y)}")
    if len(errores) > 0:
        print(errores[["nombre_real", "nombre_predicho",
                        "element", "mass", "coordination"]].to_string(index=False))

    print(f"\n✅ Validación externa completa en: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()