import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ======================================
# CONFIG
# ======================================
DATA_PATH = "/home/marazi/proyectoParam/scripts/prueba1/Preprocesamiento/output1/preprocessed_data.csv"
MODEL_PATH = "mlp_model.joblib"
SCALER_PATH = "scaler.joblib"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ======================================
# MAIN
# ======================================
def main():
    print("📥 Cargando dataset preprocesado...")

    if not Path(DATA_PATH).exists():
        raise FileNotFoundError(f"No se encontró: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # =============================
    # Filtrar clases con <2 muestras
    # =============================
    counts = df["target_atom_type"].value_counts()
    valid_classes = counts[counts >= 2].index
    df = df[df["target_atom_type"].isin(valid_classes)].copy()

    # =============================
    # Features y target
    # =============================
    X = df.drop(columns=["target_atom_type", "molecule"], errors="ignore")
    y = df["target_atom_type"]

    print("Shape X:", X.shape)
    print("Clases únicas:", y.nunique())

    # =============================
    # Train / Test
    # =============================
    print("\n✂️ Separando train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # =============================
    # Escalado
    # =============================
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # =============================
    # Modelo MLP
    # =============================
    print("\n🧠 Entrenando MLP baseline...")
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=500,
        random_state=RANDOM_STATE,
    )

    model.fit(X_train_scaled, y_train)

    # =============================
    # Evaluación
    # =============================
    print("\n📊 Evaluando modelo...")
    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n🎯 Accuracy: {acc:.4f}\n")

    print("📋 Classification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("\n🧩 Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # =============================
    # Guardar modelo
    # =============================
    Path("Modelo").mkdir(exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print(f"\n💾 Modelo guardado en: {MODEL_PATH}")
    print(f"💾 Scaler guardado en: {SCALER_PATH}")


# ======================================
if __name__ == "__main__":
    main()