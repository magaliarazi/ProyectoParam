import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight

# =========================
# CONFIG
# =========================
DATA_PATH = "/home/marazi/proyectoParam/scripts/prueba2/processed_AA/dataset_processed.csv"
OUTPUT_DIR = "model1_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["atomtype_encoded", "atomtype_label", "charge"])
y = df["atomtype_encoded"]

# =========================
# 🔥 AGRUPAR CLASES RARAS
# =========================
threshold = 5

counts = y.value_counts()
rare_classes = counts[counts < threshold].index

print("\n⚠ Clases agrupadas en OTHER:")
print(rare_classes)

y = y.replace(rare_classes, -1)

print("\n📊 Nuevo balance:")
print(y.value_counts())

# =========================
# SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# =========================
# CLASS WEIGHTS
# =========================
classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = dict(zip(classes, weights))
sample_weights = y_train.map(class_weights)

# =========================
# 🧠 MLP MODEL
# =========================
mlp = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation='relu',
    solver='adam',
    alpha=1e-3,
    batch_size=256,
    learning_rate_init=1e-3,
    max_iter=300,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=15,
    random_state=42
)

print("\n🚀 Entrenando MLP...")
mlp.fit(X_train, y_train, sample_weight=sample_weights)

y_pred_mlp = mlp.predict(X_test)

# =========================
# 🌳 RANDOM FOREST
# =========================
rf = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("\n🌳 Entrenando Random Forest...")
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

# =========================
# 📊 COMPARACIÓN
# =========================
f1_mlp = f1_score(y_test, y_pred_mlp, average="macro")
f1_rf = f1_score(y_test, y_pred_rf, average="macro")

print(f"\n🔥 F1 Macro MLP: {f1_mlp:.3f}")
print(f"🔥 F1 Macro RF:  {f1_rf:.3f}")

# =========================
# METRICS (MLP)
# =========================
print("\n📊 REPORT MLP")
report_mlp = classification_report(y_test, y_pred_mlp, zero_division=0)
print(report_mlp)

with open(os.path.join(OUTPUT_DIR, "report_mlp.txt"), "w") as f:
    f.write(report_mlp)

# =========================
# METRICS (RF)
# =========================
print("\n📊 REPORT RF")
report_rf = classification_report(y_test, y_pred_rf, zero_division=0)
print(report_rf)

with open(os.path.join(OUTPUT_DIR, "report_rf.txt"), "w") as f:
    f.write(report_rf)

# =========================
# MATRIZ CONFUSIÓN (MLP)
# =========================
cm = confusion_matrix(y_test, y_pred_mlp)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, cmap="Blues")
plt.title("Matriz de Confusión (MLP)")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_mlp.png"), dpi=300)
plt.show()

# =========================
# LOSS CURVE (MLP)
# =========================
plt.figure(figsize=(8,5))
plt.plot(mlp.loss_curve_)
plt.title("Loss Curve (MLP)")
plt.xlabel("Iteraciones")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"), dpi=300)
plt.show()

# =========================
# VALIDATION CURVE
# =========================
if hasattr(mlp, "validation_scores_"):
    plt.figure(figsize=(8,5))
    plt.plot(mlp.validation_scores_)
    plt.title("Validation Score (MLP)")
    plt.xlabel("Iteraciones")
    plt.ylabel("Score")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "validation_curve.png"), dpi=300)
    plt.show()

# =========================
# F1 POR CLASE (MLP)
# =========================
report_dict = classification_report(y_test, y_pred_mlp, output_dict=True, zero_division=0)
report_df = pd.DataFrame(report_dict).T.iloc[:-3]

plt.figure(figsize=(14,6))
sns.barplot(x=report_df.index, y=report_df["f1-score"])
plt.xticks(rotation=90)
plt.title("F1-score por clase (MLP)")
plt.ylabel("F1")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "f1_per_class.png"), dpi=300)
plt.show()

# =========================
# DISTRIBUCIÓN
# =========================
real_counts = pd.Series(y_test).value_counts().sort_index()
pred_counts = pd.Series(y_pred_mlp).value_counts().sort_index()

df_counts = pd.DataFrame({
    "Real": real_counts,
    "Predicho": pred_counts
}).fillna(0)

df_counts.plot(kind="bar", figsize=(14,6))
plt.title("Distribución: Real vs Predicho (MLP)")
plt.ylabel("Cantidad")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "distribution.png"), dpi=300)
plt.show()

# =========================
# FEATURE IMPORTANCE (RF)
# =========================
importances = rf.feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)

feat_imp.head(20).to_csv(os.path.join(OUTPUT_DIR, "top_features.csv"))

print("\n🔬 Top features (RF):")
print(feat_imp.head(10))

# =========================
# DONE
# =========================
print(f"\n✅ Todo listo. Resultados en: {OUTPUT_DIR}")