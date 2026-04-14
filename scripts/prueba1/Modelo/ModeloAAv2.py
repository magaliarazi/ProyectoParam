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
# Ajustado a la ruta de tu archivo preprocesado
DATA_PATH = "/home/marazi/proyectoParam/scripts/prueba1/Preprocesamiento/tabla_all_atom.csv"
MODEL_PATH = "mlp_model_AA.joblib"
SCALER_PATH = "scaler_AA.joblib"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ======================================
# MAIN
# ======================================
def main():
    print("📥 Cargando dataset preprocesado...")

    if not Path(DATA_PATH).exists():
        raise FileNotFoundError(f"No se encontró el archivo en la ruta: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # =============================
    # Filtrar clases con <2 muestras
    # =============================
    # Esto es vital para que train_test_split(stratify=y) no falle
    counts = df["target_atom_type"].value_counts()
    valid_classes = counts[counts >= 2].index
    df = df[df["target_atom_type"].isin(valid_classes)].copy()

    # =============================
    # Features y target
    # =============================
    # Quitamos el target y la identificación de molécula. 
    # También quitamos target_charge si queremos predecir el TIPO de átomo basándonos en geometría.
    X = df.drop(columns=["target_atom_type", "molecule", "target_charge"], errors="ignore")
    y = df["target_atom_type"]

    print("Shape X (Atributos):", X.shape)
    print("Clases únicas a predecir:", y.nunique())
    print("Atributos utilizados:", list(X.columns))

    # =============================
    # Train / Test
    # =============================
    print("\n✂️ Separando datos en entrenamiento y prueba...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # =============================
    # Escalado
    # =============================
    # 
    # Escalamos para que variables como la masa no dominen sobre los ángulos o distancias
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # =============================
    # Modelo MLP (Red Neuronal)
    # =============================
    # 
    print("\n🧠 Entrenando Perceptrón Multicapa (MLP)...")
    model = MLPClassifier(
        hidden_layer_sizes=(100, 50), # Configuración recomendada para tu tesis
        activation="relu",
        max_iter=1000,                # Más iteraciones para asegurar convergencia
        random_state=RANDOM_STATE,
    )

    model.fit(X_train_scaled, y_train)

    # =============================
    # Evaluación
    # =============================
    print("\n📊 Evaluando modelo...")
    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n🎯 Accuracy (Exactitud): {acc:.4f}\n")

    print("📋 Reporte de Clasificación:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("\n🧩 Matriz de Confusión:")
    print(confusion_matrix(y_test, y_pred))

    # =============================
    # Guardar modelo
    # =============================
    # Crear carpeta si no existe
    Path("Modelo").mkdir(exist_ok=True)

    joblib.dump(model, f"Modelo/{MODEL_PATH}")
    joblib.dump(scaler, f"Modelo/{SCALER_PATH}")

    print(f"\n💾 Modelo guardado en: Modelo/{MODEL_PATH}")
    print(f"💾 Scaler guardado en: Modelo/{SCALER_PATH}")


# ======================================
if __name__ == "__main__":
    main()
