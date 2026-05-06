import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# =========================
# CONFIG
# =========================
DATA_PATH = "/home/marazi/proyectoParam/scripts/prueba2/processed_AA/dataset_processed.csv"
OUTPUT_DIR = "rf_results"

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
# 🌳 RANDOM FOREST
# =========================
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("\n🌳 Entrenando Random Forest...")
rf.fit(X_train, y_train)

# =========================
# PREDICT
# =========================
y_pred = rf.predict(X_test)

# =========================
# METRICS
# =========================
print("\n📊 REPORT")
report = classification_report(y_test, y_pred, zero_division=0)
print(report)

f1 = f1_score(y_test, y_pred, average="macro")
print(f"\n🔥 F1 Macro (test): {f1:.3f}")

with open(os.path.join(OUTPUT_DIR, "report.txt"), "w") as f:
    f.write(report)

# =========================
# 📊 VALIDACIÓN CRUZADA
# =========================
print("\n🔁 Validación cruzada (5 folds)...")

cv_scores = cross_val_score(
    rf,
    X,
    y,
    cv=5,
    scoring="f1_macro",
    n_jobs=-1
)

print("\n📊 F1 por fold:", cv_scores)
print(f"📊 Promedio: {cv_scores.mean():.3f}")
print(f"📊 Desvío:   {cv_scores.std():.3f}")

# guardar resultados
with open(os.path.join(OUTPUT_DIR, "cv_scores.txt"), "w") as f:
    f.write(f"Scores: {cv_scores}\n")
    f.write(f"Mean: {cv_scores.mean()}\n")
    f.write(f"Std: {cv_scores.std()}\n")

# =========================
# MATRIZ DE CONFUSIÓN
# =========================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, cmap="Blues")
plt.title("Matriz de Confusión (RF)")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=300)
plt.show()

# =========================
# F1 POR CLASE
# =========================
report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
report_df = pd.DataFrame(report_dict).T.iloc[:-3]

plt.figure(figsize=(14,6))
sns.barplot(x=report_df.index, y=report_df["f1-score"])
plt.xticks(rotation=90)
plt.title("F1-score por clase (RF)")
plt.ylabel("F1")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "f1_per_class.png"), dpi=300)
plt.show()

# =========================
# FEATURE IMPORTANCE
# =========================
importances = rf.feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)

feat_imp.head(20).to_csv(os.path.join(OUTPUT_DIR, "top_features.csv"))

print("\n🔬 Top features:")
print(feat_imp.head(10))

#===================
# ANALISIS DE ERRORES
#====================
# =========================
# 🔍 ANÁLISIS DE ERRORES
# =========================
errors = pd.DataFrame({
    "real": y_test,
    "pred": y_pred
})

errors = errors[errors["real"] != errors["pred"]]

print(f"\n❌ Total errores: {len(errors)}")

# =========================
# 🔥 1. MATRIZ DE CONFUSIÓN NORMALIZADA
# =========================
cm = confusion_matrix(y_test, y_pred, normalize='true')

plt.figure(figsize=(12,10))
sns.heatmap(cm, cmap="Reds")
plt.title("Confusión normalizada (errores relativos)")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_normalized.png"), dpi=300)
plt.show()

# =========================
# 🔥 2. PARES MÁS CONFUNDIDOS
# =========================
pairs = errors.groupby(["real", "pred"]).size().sort_values(ascending=False)

print("\n🔁 Confusiones más frecuentes:")
print(pairs.head(10))

pairs.head(10).to_csv(os.path.join(OUTPUT_DIR, "top_confusions.csv"))

# =========================
# 🔥 3. ERROR POR CLASE
# =========================
error_rate = (y_test != y_pred).groupby(y_test).mean()

print("\n📊 Error rate por clase:")
print(error_rate.sort_values(ascending=False))

error_rate.to_csv(os.path.join(OUTPUT_DIR, "error_per_class.csv"))

# =========================
# 🔥 4. FEATURE ANALYSIS EN ERRORES
# =========================
X_test_errors = X_test.loc[errors.index]

print("\n🔬 Promedio features en errores:")
print(X_test_errors.mean().sort_values(ascending=False).head(10))

X_test_errors.mean().to_csv(os.path.join(OUTPUT_DIR, "error_feature_means.csv"))

# =========================
# 🔥 5. COMPARACIÓN ERROR vs CORRECTO
# =========================
correct = y_test == y_pred

mean_correct = X_test[correct].mean()
mean_error = X_test[~correct].mean()

diff = (mean_error - mean_correct).sort_values(ascending=False)

print("\n⚖ Diferencia (error - correcto):")
print(diff.head(10))

diff.to_csv(os.path.join(OUTPUT_DIR, "feature_diff_error_vs_correct.csv"))

print("\n✅ Análisis de errores completo guardado")

# =========================
# DONE
# =========================
print(f"\n✅ Todo listo. Resultados en: {OUTPUT_DIR}")