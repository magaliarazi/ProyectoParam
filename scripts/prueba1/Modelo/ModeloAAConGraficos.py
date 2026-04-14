import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)

# ======================================
# CONFIG
# ======================================
# Cambiá AA por UA para correr el análisis de UA
DATA_PATH    = "/home/marazi/proyectoParam/scripts/prueba1/Preprocesamiento/tabla_all_atom.csv"
MODELO_NOMBRE = "AA"   

TEST_SIZE    = 0.2
RANDOM_STATE = 42
OUTPUT_DIR   = Path(f"analisis_errores_{MODELO_NOMBRE}")

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

    print(f"📥 Cargando dataset {MODELO_NOMBRE}...")
    df = pd.read_csv(DATA_PATH)

    # Filtrar clases con <2 muestras
    counts = df["target_atom_type"].value_counts()
    valid_classes = counts[counts >= 2].index
    df = df[df["target_atom_type"].isin(valid_classes)].copy()

    X = df.drop(columns=["target_atom_type", "molecule", "target_charge"], errors="ignore")
    y = df["target_atom_type"]

    print(f"Shape X: {X.shape} | Clases: {y.nunique()}")

    # Train / Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # Escalado
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    #Smote
    smote = SMOTE(
    random_state=RANDOM_STATE,
    k_neighbors=1
    )
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

    # ======================================
    # Modelo con early stopping
    # ======================================
    print("\n🧠 Entrenando MLP con early stopping...")
    model = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation="relu",
        max_iter=1000,
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
    )
    model.fit(X_train_res, y_train_res)

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n🎯 Accuracy: {acc:.4f}")
    print(f"   Épocas entrenadas: {len(model.loss_curve_)}")
    print("\n📋 Reporte:")
    print(classification_report(y_test, y_pred, zero_division=0))

    clases_presentes = sorted(y_test.unique())
    nombres = [LABEL_NAMES.get(c, str(c)) for c in clases_presentes]

    # ======================================
    # 1. CURVA DE LOSS
    # ======================================
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(model.loss_curve_, label="Train loss",
            color="steelblue", linewidth=2)
    ax.plot(model.validation_scores_, label="Validation score",
            color="tomato", linewidth=2, linestyle="--")
    ax.set_xlabel("Época", fontsize=12)
    ax.set_ylabel("Valor", fontsize=12)
    ax.set_title(f"Curva de entrenamiento — {MODELO_NOMBRE}", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"loss_curve_{MODELO_NOMBRE}.png", dpi=150)
    plt.close()
    print(f"\n📈 Loss curve → {OUTPUT_DIR}/loss_curve_{MODELO_NOMBRE}.png")

    # ======================================
    # 2. MATRIZ DE CONFUSIÓN
    # ======================================
    cm = confusion_matrix(y_test, y_pred, labels=clases_presentes)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=nombres, yticklabels=nombres,
                linewidths=0.5, ax=ax)
    ax.set_xlabel("Predicho", fontsize=12)
    ax.set_ylabel("Real", fontsize=12)
    ax.set_title(f"Matriz de Confusión — {MODELO_NOMBRE}", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"confusion_matrix_{MODELO_NOMBRE}.png", dpi=150)
    plt.close()
    print(f"📊 Confusion matrix → {OUTPUT_DIR}/confusion_matrix_{MODELO_NOMBRE}.png")

    # ======================================
    # 3. DISTRIBUCIÓN DE CLASES
    # ======================================
    class_counts = y.value_counts().sort_index()
    class_names  = [LABEL_NAMES.get(c, str(c)) for c in class_counts.index]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(class_names, class_counts.values,
                  color="steelblue", edgecolor="white")
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_xlabel("Tipo de átomo", fontsize=12)
    ax.set_ylabel("Cantidad de ejemplos", fontsize=12)
    ax.set_title(f"Distribución de clases — {MODELO_NOMBRE}", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"distribucion_clases_{MODELO_NOMBRE}.png", dpi=150)
    plt.close()
    print(f"📊 Distribución → {OUTPUT_DIR}/distribucion_clases_{MODELO_NOMBRE}.png")

    # ======================================
    # 4. F1 POR CLASE
    # ======================================
    f1_scores = f1_score(y_test, y_pred, labels=clases_presentes,
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
    ax.set_title(f"F1-score por clase — {MODELO_NOMBRE}", fontsize=14)
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"f1_por_clase_{MODELO_NOMBRE}.png", dpi=150)
    plt.close()
    print(f"📊 F1 por clase → {OUTPUT_DIR}/f1_por_clase_{MODELO_NOMBRE}.png")

    # ======================================
    # 5. ERRORES — CSV
    # ======================================
    X_test_df = X_test.copy()
    X_test_df["real"]            = y_test.values
    X_test_df["predicho"]        = y_pred
    X_test_df["nombre_real"]     = X_test_df["real"].map(LABEL_NAMES)
    X_test_df["nombre_predicho"] = X_test_df["predicho"].map(LABEL_NAMES)
    X_test_df["correcto"]        = y_test.values == y_pred

    errores = X_test_df[~X_test_df["correcto"]].copy()
    print(f"\n❌ Errores: {len(errores)} de {len(y_test)}")
    if len(errores) > 0:
        print(errores[["nombre_real", "nombre_predicho", "element",
                        "mass", "coordination", "neighbor_config"]].to_string(index=False))

    errores.to_csv(OUTPUT_DIR / f"errores_{MODELO_NOMBRE}.csv", index=False)
    print(f"💾 Errores → {OUTPUT_DIR}/errores_{MODELO_NOMBRE}.csv")

    # Guardar modelo y scaler
    Path("Modelo").mkdir(exist_ok=True)
    joblib.dump(model,  f"Modelo/mlp_model_{MODELO_NOMBRE}.joblib")
    joblib.dump(scaler, f"Modelo/scaler_{MODELO_NOMBRE}.joblib")
    print(f"\n💾 Modelo guardado en: Modelo/mlp_model_{MODELO_NOMBRE}.joblib")
    print(f"💾 Scaler guardado en: Modelo/scaler_{MODELO_NOMBRE}.joblib")


    print(f"\n✅ Análisis completo en: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()