"""
model_AA.py
-----------
Entrena y evalúa el clasificador Random Forest (E3) para el dataset All-Atom (AA).

Nota: la regresión de charge en AA usa TabTransformer (tabtransformer_atomtype_charge.py).
Este script entrena y evalúa únicamente el clasificador de atomtype.

Salidas:
  results_AA/
    ├── clf_AA.joblib                  modelo de clasificación (atomtype)
    ├── metrics_AA.json                métricas completas
    ├── fig_atomtype_distribution.png
    ├── fig_confusion_matrix.png
    ├── fig_feature_importance_clf.png
    └── fig_crossval_scores.png

Uso:
    python AA.py --input aa_clean.csv --output_dir results_AA/
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)
from sklearn.model_selection import (
    StratifiedKFold, cross_validate, train_test_split,
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"]  = 11

LABEL = "AA"

# =============================================================================
# CONFIG
# =============================================================================

TARGET_CLF = "atomtype"
TARGET_REG = "charge"
TEST_SIZE   = 0.20
RANDOM_SEED = 42
CV_FOLDS    = 5

# Hiperparámetros E3 — seleccionados como mejor configuración
# n_estimators=500:    mayor estabilidad en las predicciones
# max_depth=None:      sin límite de profundidad
# min_samples_leaf=2:  evita sobreajuste en hojas con pocas instancias
# max_features="sqrt": mayor diversidad entre árboles
# class_weight="balanced": compensa el desbalance severo (ratio 253x en AA)
RF_CLF_PARAMS = dict(
    n_estimators     = 500,
    max_depth        = None,
    min_samples_leaf = 2,
    max_features     = "sqrt",
    class_weight     = "balanced",
    random_state     = RANDOM_SEED,
    n_jobs           = -1,
)

TOP_N_FEATURES = 20


# =============================================================================
# HELPERS
# =============================================================================

def save(fig, out_dir, name):
    path = out_dir / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path.name}")


def section(title):
    print(f"\n{'='*55}\n  {title}\n{'='*55}")


# =============================================================================
# CARGA Y SPLIT
# =============================================================================

def load_and_split(csv_path: Path):
    df = pd.read_csv(csv_path)
    print(f"Dataset cargado: {df.shape}")
    print(f"  Clases atomtype: {df[TARGET_CLF].nunique()}")

    feature_cols = [c for c in df.columns if c not in [TARGET_CLF, TARGET_REG]]
    X     = df[feature_cols]
    y_clf = df[TARGET_CLF]
    y_reg = df[TARGET_REG]

    MIN_SAMPLES = 2
    counts  = y_clf.value_counts()
    removed = sorted(counts[counts < MIN_SAMPLES].index.tolist())
    mask    = y_clf.isin(counts[counts >= MIN_SAMPLES].index)
    X       = X[mask].reset_index(drop=True)
    y_clf   = y_clf[mask].reset_index(drop=True)
    y_reg   = y_reg[mask].reset_index(drop=True)
    print(f"  Clases eliminadas (1 instancia): {removed}")
    print(f"  Átomos restantes:  {len(X)}")

    non_numeric = X.select_dtypes(include=["object"]).columns.tolist()
    print(f"  Columnas no numéricas: {non_numeric}")

    X_train, X_test, y_clf_train, y_clf_test, y_reg_train, y_reg_test = \
        train_test_split(
            X, y_clf, y_reg,
            test_size    = TEST_SIZE,
            random_state = RANDOM_SEED,
            stratify     = y_clf,
        )

    print(f"\n  Train: {len(X_train)} átomos ({len(X_train)/len(X)*100:.0f}%)")
    print(f"  Test:  {len(X_test)} átomos ({len(X_test)/len(X)*100:.0f}%)")

    return X, X_train, X_test, y_clf, y_clf_train, y_clf_test, \
           y_reg, y_reg_train, y_reg_test, feature_cols


# =============================================================================
# VISUALIZACIÓN
# =============================================================================

def plot_class_distribution(y_clf, y_clf_train, y_clf_test, out_dir):
    section("Distribución de clases")
    counts_full  = y_clf.value_counts()
    counts_train = y_clf_train.value_counts().reindex(counts_full.index, fill_value=0)
    counts_test  = y_clf_test.value_counts().reindex(counts_full.index,  fill_value=0)
    x = np.arange(len(counts_full))
    w = 0.35
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - w/2, counts_train.values, w, label="Train", color="#378ADD")
    ax.bar(x + w/2, counts_test.values,  w, label="Test",  color="#D85A30")
    ax.set_xticks(x)
    ax.set_xticklabels(counts_full.index, rotation=45, ha="right")
    ax.set_xlabel("atomtype")
    ax.set_ylabel("Frecuencia")
    ax.set_title(f"Distribución de clases — {LABEL} (Train vs Test)")
    ax.legend()
    plt.tight_layout()
    save(fig, out_dir, "fig_atomtype_distribution.png")


# =============================================================================
# CLASIFICACIÓN
# =============================================================================

def train_classifier(X_train, y_train):
    section("Entrenando clasificador (atomtype) — RF E3")
    clf = RandomForestClassifier(**RF_CLF_PARAMS)
    clf.fit(X_train, y_train)
    print("  ✓ Clasificador entrenado")
    return clf


def evaluate_classifier(clf, X_train, X_test, y_train, y_test,
                         X_full, y_full, feature_cols, out_dir):
    section("Evaluación — Clasificación")
    y_pred_train = clf.predict(X_train)
    y_pred_test  = clf.predict(X_test)
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test  = accuracy_score(y_test,  y_pred_test)
    f1_macro  = f1_score(y_test, y_pred_test, average="macro",    zero_division=0)
    f1_w      = f1_score(y_test, y_pred_test, average="weighted", zero_division=0)
    print(f"  Accuracy train:  {acc_train:.4f}")
    print(f"  Accuracy test:   {acc_test:.4f}")
    print(f"  F1 macro:        {f1_macro:.4f}")
    print(f"  F1 weighted:     {f1_w:.4f}")
    print()
    print(classification_report(y_test, y_pred_test, zero_division=0))

    section("Cross-validation — Clasificación")
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    cv_results = cross_validate(
        RandomForestClassifier(**RF_CLF_PARAMS), X_full, y_full,
        cv=skf, scoring=["accuracy", "f1_macro", "f1_weighted"],
        return_train_score=True, n_jobs=-1,
    )
    cv_acc = cv_results["test_accuracy"]
    cv_f1m = cv_results["test_f1_macro"]
    cv_f1w = cv_results["test_f1_weighted"]
    print(f"  CV Accuracy:    {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
    print(f"  CV F1 macro:    {cv_f1m.mean():.4f} ± {cv_f1m.std():.4f}")
    print(f"  CV F1 weighted: {cv_f1w.mean():.4f} ± {cv_f1w.std():.4f}")

    classes = sorted(y_full.unique())
    cm      = confusion_matrix(y_test, y_pred_test, labels=classes)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes,
                linewidths=0.5, ax=ax)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusión — {LABEL} (test set)")
    plt.tight_layout()
    save(fig, out_dir, "fig_confusion_matrix.png")

    importances = pd.Series(clf.feature_importances_, index=feature_cols)
    top = importances.nlargest(TOP_N_FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top.index, top.values, color=sns.color_palette("muted", len(top)))
    ax.set_xlabel("Importancia (Mean Decrease Impurity)")
    ax.set_title(f"Feature importance — Clasificación {LABEL}\n(top {TOP_N_FEATURES})")
    plt.tight_layout()
    save(fig, out_dir, "fig_feature_importance_clf.png")

    folds = np.arange(1, CV_FOLDS + 1)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, scores, title, color in zip(
        axes,
        [cv_acc, cv_f1m, cv_f1w],
        ["Accuracy", "F1 macro", "F1 weighted"],
        ["#378ADD", "#1D9E75", "#534AB7"],
    ):
        ax.bar(folds, scores, color=color, alpha=0.8)
        ax.axhline(scores.mean(), color="red", linestyle="--",
                   label=f"Media: {scores.mean():.3f}")
        ax.set_xticks(folds)
        ax.set_xlabel("Fold")
        ax.set_ylabel(title)
        ax.set_title(f"{title} por fold")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=9)
    plt.suptitle(f"Cross-validation ({CV_FOLDS} folds) — Clasificación {LABEL}", y=1.02)
    plt.tight_layout()
    save(fig, out_dir, "fig_crossval_scores.png")

    return {
        "accuracy_train":      round(acc_train, 4),
        "accuracy_test":       round(acc_test,  4),
        "f1_macro":            round(f1_macro,  4),
        "f1_weighted":         round(f1_w,      4),
        "cv_accuracy_mean":    round(float(cv_acc.mean()), 4),
        "cv_accuracy_std":     round(float(cv_acc.std()),  4),
        "cv_f1_macro_mean":    round(float(cv_f1m.mean()), 4),
        "cv_f1_macro_std":     round(float(cv_f1m.std()),  4),
        "cv_f1_weighted_mean": round(float(cv_f1w.mean()), 4),
        "cv_f1_weighted_std":  round(float(cv_f1w.std()),  4),
        "classification_report": classification_report(
            y_test, y_pred_test, zero_division=0, output_dict=True
        ),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=f"Clasificador RF E3 — {LABEL}")
    parser.add_argument("--input",      required=True, help="CSV limpio (aa_clean.csv)")
    parser.add_argument("--output_dir", default=f"results_{LABEL}/")
    args    = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  CLASIFICADOR RF E3 — {LABEL}")
    print(f"  Nota: regresión de charge → TabTransformer (script separado)")
    print(f"{'='*55}")

    (X, X_train, X_test,
     y_clf, y_clf_train, y_clf_test,
     y_reg, y_reg_train, y_reg_test,
     feature_cols) = load_and_split(Path(args.input))

    plot_class_distribution(y_clf, y_clf_train, y_clf_test, out_dir)

    clf         = train_classifier(X_train, y_clf_train)
    metrics_clf = evaluate_classifier(
        clf, X_train, X_test, y_clf_train, y_clf_test,
        X, y_clf, feature_cols, out_dir
    )

    section("Guardando modelo")
    clf_path = out_dir / f"clf_{LABEL}.joblib"
    dump(clf, clf_path)
    print(f"  → {clf_path.name}")

    metrics = {
        "dataset":        LABEL,
        "experimento":    "E3",
        "modelo":         "RandomForestClassifier",
        "nota":           "Regresión de charge en AA → TabTransformer (script separado)",
        "n_train":        len(X_train),
        "n_test":         len(X_test),
        "n_features":     len(feature_cols),
        "test_size":      TEST_SIZE,
        "cv_folds":       CV_FOLDS,
        "rf_clf_params":  RF_CLF_PARAMS,
        "classification": metrics_clf,
    }
    metrics_path = out_dir / f"metrics_{LABEL}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  → {metrics_path.name}")

    section("RESUMEN FINAL")
    print(f"  Clasificación (atomtype):")
    print(f"    Accuracy test:   {metrics_clf['accuracy_test']:.4f}")
    print(f"    F1 macro test:   {metrics_clf['f1_macro']:.4f}")
    print(f"    CV Accuracy:     {metrics_clf['cv_accuracy_mean']:.4f} "
          f"± {metrics_clf['cv_accuracy_std']:.4f}")
    print(f"\n  ⚠ Regresión de charge: correr tabtransformer_atomtype_charge.py")
    print(f"\n✔ Resultados guardados en: {out_dir}")


if __name__ == "__main__":
    main()