"""
model_AA_compare.py
-------------------
Compara múltiples configuraciones de modelos para el dataset AA.

Experimentos:
  EXP 1 — RF baseline (igual al original, para referencia)
  EXP 2 — RF con hiperparámetros más conservadores (reduce overfitting)
  EXP 3 — RF con max_depth=None (árboles completos)
  EXP 4 — GradientBoosting (regresor más robusto para datasets pequeños)
  EXP 5 — XGBoost (si está instalado, si no se saltea)
  EXP 6 — RF + feature selection (top features por importancia)
  EXP 7 — Stacking: RF + GBR con meta-learner Ridge

Para cada experimento se reportan:
  - Métricas de test (split 80/20 estratificado)
  - CV 5-fold (mismo esquema que el original para comparar)
  - Tabla resumen al final

Uso:
    python model_AA_compare.py --input processed/aa_clean.csv --output_dir results_compare/

Dependencias:
    pip install pandas numpy scikit-learn matplotlib seaborn joblib
    pip install xgboost  (opcional)
"""

import argparse
import json
import warnings
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import dump
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
    StackingClassifier, StackingRegressor,
)
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, cross_validate, train_test_split,
)
from sklearn.svm import SVR

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"]  = 11

LABEL        = "AA"
TARGET_CLF   = "atomtype"
TARGET_REG   = "charge"
TEST_SIZE    = 0.20
RANDOM_SEED  = 42
CV_FOLDS     = 5
TOP_FEATURES = 20   # para EXP 6

# =============================================================================
# HELPERS
# =============================================================================

def save(fig, out_dir, name):
    path = out_dir / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"    → {path.name}")

def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")

def divider(exp_name):
    print(f"\n{'─'*60}")
    print(f"  {exp_name}")
    print(f"{'─'*60}")

# =============================================================================
# CARGA Y SPLIT (igual al original)
# =============================================================================

def load_and_split(csv_path: Path):
    df = pd.read_csv(csv_path)
    print(f"Dataset cargado: {df.shape}")
    print(f"  Clases atomtype: {df[TARGET_CLF].nunique()}")
    print(f"  Rango charge:    [{df[TARGET_REG].min():.3f}, {df[TARGET_REG].max():.3f}]")

    feature_cols = [c for c in df.columns if c not in [TARGET_CLF, TARGET_REG]]
    X     = df[feature_cols]
    y_clf = df[TARGET_CLF]
    y_reg = df[TARGET_REG]

    # Eliminar clases con < 2 instancias
    counts  = y_clf.value_counts()
    removed = sorted(counts[counts < 2].index.tolist())
    mask    = y_clf.isin(counts[counts >= 2].index)
    X       = X[mask].reset_index(drop=True)
    y_clf   = y_clf[mask].reset_index(drop=True)
    y_reg   = y_reg[mask].reset_index(drop=True)
    print(f"  Clases eliminadas (1 instancia): {removed}")
    print(f"  Átomos restantes: {len(X)}")

    # Eliminar columnas no numéricas
    non_num = X.select_dtypes(include=["object"]).columns.tolist()
    if non_num:
        print(f"  ⚠ Eliminando columnas no numéricas: {non_num}")
        X = X.drop(columns=non_num)
        feature_cols = [c for c in feature_cols if c not in non_num]

    X_train, X_test, y_clf_train, y_clf_test, y_reg_train, y_reg_test = \
        train_test_split(
            X, y_clf, y_reg,
            test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y_clf,
        )

    print(f"  Train: {len(X_train)} átomos | Test: {len(X_test)} átomos")
    return X, X_train, X_test, y_clf, y_clf_train, y_clf_test, \
           y_reg, y_reg_train, y_reg_test, feature_cols

# =============================================================================
# EVALUACIÓN GENÉRICA (usada por todos los experimentos)
# =============================================================================

def eval_clf(clf, X_train, X_test, y_train, y_test, X_full, y_full, label):
    t0 = time()
    clf.fit(X_train, y_train)
    t_fit = time() - t0

    y_pred = clf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1m    = f1_score(y_test, y_pred, average="macro",    zero_division=0)
    f1w    = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    cv  = cross_validate(clf, X_full, y_full, cv=skf,
                         scoring=["accuracy", "f1_macro", "f1_weighted"],
                         n_jobs=-1)
    cv_acc = cv["test_accuracy"]
    cv_f1m = cv["test_f1_macro"]
    cv_f1w = cv["test_f1_weighted"]

    print(f"  [{label}] Acc test={acc:.4f} | F1-macro={f1m:.4f} | "
          f"CV Acc={cv_acc.mean():.4f}±{cv_acc.std():.4f} | fit={t_fit:.1f}s")

    return {
        "accuracy_test":       round(acc, 4),
        "f1_macro":            round(f1m, 4),
        "f1_weighted":         round(f1w, 4),
        "cv_accuracy_mean":    round(float(cv_acc.mean()), 4),
        "cv_accuracy_std":     round(float(cv_acc.std()),  4),
        "cv_f1_macro_mean":    round(float(cv_f1m.mean()), 4),
        "cv_f1_macro_std":     round(float(cv_f1m.std()),  4),
        "cv_f1_weighted_mean": round(float(cv_f1w.mean()), 4),
        "cv_f1_weighted_std":  round(float(cv_f1w.std()),  4),
        "fit_time_s":          round(t_fit, 2),
    }, clf


def eval_reg(reg, X_train, X_test, y_train, y_test, X_full, y_full, label):
    t0 = time()
    reg.fit(X_train, y_train)
    t_fit = time() - t0

    y_pred  = reg.predict(X_test)
    mae     = mean_absolute_error(y_test, y_pred)
    rmse    = np.sqrt(mean_squared_error(y_test, y_pred))
    r2      = r2_score(y_test, y_pred)

    cv_mae  = -cross_val_score(reg, X_full, y_full, cv=CV_FOLDS,
                                scoring="neg_mean_absolute_error", n_jobs=-1)
    cv_r2   = cross_val_score(reg, X_full, y_full, cv=CV_FOLDS,
                               scoring="r2", n_jobs=-1)

    print(f"  [{label}] R²={r2:.4f} | MAE={mae:.4f} | "
          f"CV R²={cv_r2.mean():.4f}±{cv_r2.std():.4f} | fit={t_fit:.1f}s")

    return {
        "mae_test":     round(mae,  4),
        "rmse_test":    round(rmse, 4),
        "r2_test":      round(r2,   4),
        "cv_mae_mean":  round(float(cv_mae.mean()), 4),
        "cv_mae_std":   round(float(cv_mae.std()),  4),
        "cv_r2_mean":   round(float(cv_r2.mean()),  4),
        "cv_r2_std":    round(float(cv_r2.std()),   4),
        "fit_time_s":   round(t_fit, 2),
    }, reg

# =============================================================================
# EXPERIMENTOS
# =============================================================================

def run_experiments(X, X_train, X_test,
                    y_clf, y_clf_train, y_clf_test,
                    y_reg, y_reg_train, y_reg_test,
                    feature_cols):

    results = {}

    # ------------------------------------------------------------------
    # EXP 1 — RF baseline (igual al original)
    # ------------------------------------------------------------------
    divider("EXP 1 — RF baseline (igual al original, referencia)")
    clf1 = RandomForestClassifier(
        n_estimators=150, max_depth=10, min_samples_leaf=1,
        class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1)
    reg1 = RandomForestRegressor(
        n_estimators=150, max_depth=10, min_samples_leaf=1,
        random_state=RANDOM_SEED, n_jobs=-1)

    m_clf1, _ = eval_clf(clf1, X_train, X_test, y_clf_train, y_clf_test, X, y_clf, "CLF")
    m_reg1, _ = eval_reg(reg1, X_train, X_test, y_reg_train, y_reg_test, X, y_reg, "REG")
    results["EXP1_RF_baseline"] = {"clf": m_clf1, "reg": m_reg1,
        "description": "RF original: n=150, depth=10, leaf=1"}

    # ------------------------------------------------------------------
    # EXP 2 — RF conservador (reduce overfitting en regresión)
    # min_samples_leaf más alto → árboles menos profundos, mejor generalización
    # ------------------------------------------------------------------
    divider("EXP 2 — RF conservador (min_samples_leaf=5, depth=None)")
    clf2 = RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_leaf=5,
        class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1)
    reg2 = RandomForestRegressor(
        n_estimators=200, max_depth=None, min_samples_leaf=5,
        random_state=RANDOM_SEED, n_jobs=-1)

    m_clf2, _ = eval_clf(clf2, X_train, X_test, y_clf_train, y_clf_test, X, y_clf, "CLF")
    m_reg2, _ = eval_reg(reg2, X_train, X_test, y_reg_train, y_reg_test, X, y_reg, "REG")
    results["EXP2_RF_conservador"] = {"clf": m_clf2, "reg": m_reg2,
        "description": "RF: n=200, depth=None, leaf=5 (menos overfitting)"}

    # ------------------------------------------------------------------
    # EXP 3 — RF con más árboles y max_features reducido
    # max_features="sqrt" → cada árbol ve menos features, más diversidad
    # ------------------------------------------------------------------
    divider("EXP 3 — RF con n=500 y max_features='sqrt'")
    clf3 = RandomForestClassifier(
        n_estimators=500, max_depth=None, min_samples_leaf=2,
        max_features="sqrt", class_weight="balanced",
        random_state=RANDOM_SEED, n_jobs=-1)
    reg3 = RandomForestRegressor(
        n_estimators=500, max_depth=None, min_samples_leaf=2,
        max_features="sqrt", random_state=RANDOM_SEED, n_jobs=-1)

    m_clf3, _ = eval_clf(clf3, X_train, X_test, y_clf_train, y_clf_test, X, y_clf, "CLF")
    m_reg3, _ = eval_reg(reg3, X_train, X_test, y_reg_train, y_reg_test, X, y_reg, "REG")
    results["EXP3_RF_sqrt"] = {"clf": m_clf3, "reg": m_reg3,
        "description": "RF: n=500, depth=None, leaf=2, max_features=sqrt"}

    # ------------------------------------------------------------------
    # EXP 4 — GradientBoosting
    # Construye árboles secuencialmente corrigiendo errores anteriores.
    # Generalmente supera a RF en regresión con datasets pequeños.
    # ------------------------------------------------------------------
    divider("EXP 4 — GradientBoosting")
    clf4 = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        min_samples_leaf=3, subsample=0.8, random_state=RANDOM_SEED)
    reg4 = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        min_samples_leaf=3, subsample=0.8, random_state=RANDOM_SEED)

    m_clf4, _ = eval_clf(clf4, X_train, X_test, y_clf_train, y_clf_test, X, y_clf, "CLF")
    m_reg4, _ = eval_reg(reg4, X_train, X_test, y_reg_train, y_reg_test, X, y_reg, "REG")
    results["EXP4_GradientBoosting"] = {"clf": m_clf4, "reg": m_reg4,
        "description": "GBR/GBC: n=200, lr=0.05, depth=4, subsample=0.8"}

    # ------------------------------------------------------------------
    # EXP 5 — XGBoost (si está instalado)
    # ------------------------------------------------------------------
    divider("EXP 5 — XGBoost")
    try:
        from xgboost import XGBClassifier, XGBRegressor
        clf5 = XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="mlogloss",
            random_state=RANDOM_SEED, n_jobs=-1, verbosity=0)
        reg5 = XGBRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_SEED, n_jobs=-1, verbosity=0)

        # XGBoost necesita clases numéricas para clasificación
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_clf_enc       = le.fit_transform(y_clf)
        y_clf_train_enc = le.transform(y_clf_train)
        y_clf_test_enc  = le.transform(y_clf_test)

        m_clf5, _ = eval_clf(clf5, X_train, X_test,
                              y_clf_train_enc, y_clf_test_enc,
                              X, y_clf_enc, "CLF")
        m_reg5, _ = eval_reg(reg5, X_train, X_test, y_reg_train, y_reg_test, X, y_reg, "REG")
        results["EXP5_XGBoost"] = {"clf": m_clf5, "reg": m_reg5,
            "description": "XGBoost: n=200, lr=0.05, depth=4, subsample=0.8"}
        print("  ✓ XGBoost disponible")
    except ImportError:
        print("  ⚠ XGBoost no instalado. Saltando EXP 5.")
        print("    Para instalarlo: pip install xgboost")
        results["EXP5_XGBoost"] = {"clf": None, "reg": None,
            "description": "XGBoost no instalado"}

    # ------------------------------------------------------------------
    # EXP 6 — RF + feature selection
    # Entrena RF, selecciona las top features por importancia, reentrena.
    # Con 89 features para ~400 átomos hay riesgo de ruido.
    # ------------------------------------------------------------------
    divider(f"EXP 6 — RF + feature selection (top {TOP_FEATURES} features)")

    # Selección para regresión
    selector_reg = SelectFromModel(
        RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1),
        max_features=TOP_FEATURES, threshold=-np.inf
    )
    selector_reg.fit(X_train, y_reg_train)
    X_train_sel = selector_reg.transform(X_train)
    X_test_sel  = selector_reg.transform(X_test)
    X_full_sel  = selector_reg.transform(X)
    sel_features = np.array(feature_cols)[selector_reg.get_support()].tolist()
    print(f"  Features seleccionadas ({len(sel_features)}): {sel_features[:5]}...")

    reg6 = RandomForestRegressor(
        n_estimators=200, max_depth=None, min_samples_leaf=3,
        random_state=RANDOM_SEED, n_jobs=-1)
    m_reg6, _ = eval_reg(reg6,
                          X_train_sel, X_test_sel,
                          y_reg_train, y_reg_test,
                          X_full_sel, y_reg, "REG")

    # Para clasificación usamos las mismas features
    clf6 = RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_leaf=2,
        class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1)
    m_clf6, _ = eval_clf(clf6,
                          X_train_sel, X_test_sel,
                          y_clf_train, y_clf_test,
                          X_full_sel, y_clf, "CLF")

    results["EXP6_RF_feature_selection"] = {
        "clf": m_clf6, "reg": m_reg6,
        "description": f"RF con top-{TOP_FEATURES} features por importancia",
        "selected_features": sel_features,
    }

    # ------------------------------------------------------------------
    # EXP 7 — Stacking (RF + GBR con meta-learner Ridge/Logistic)
    # Combina las predicciones de múltiples modelos con un meta-modelo.
    # ------------------------------------------------------------------
    divider("EXP 7 — Stacking: RF + GBR + meta-learner")

    estimators_clf = [
        ("rf",  RandomForestClassifier(n_estimators=150, max_depth=10,
                    class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1)),
        ("gbr", GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                    max_depth=3, random_state=RANDOM_SEED)),
    ]
    clf7 = StackingClassifier(
        estimators=estimators_clf,
        final_estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
        cv=3, n_jobs=-1,
    )

    estimators_reg = [
        ("rf",  RandomForestRegressor(n_estimators=150, max_depth=10,
                    random_state=RANDOM_SEED, n_jobs=-1)),
        ("gbr", GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                    max_depth=3, random_state=RANDOM_SEED)),
    ]
    reg7 = StackingRegressor(
        estimators=estimators_reg,
        final_estimator=Ridge(alpha=1.0),
        cv=3, n_jobs=-1,
    )

    m_clf7, _ = eval_clf(clf7, X_train, X_test, y_clf_train, y_clf_test, X, y_clf, "CLF")
    m_reg7, _ = eval_reg(reg7, X_train, X_test, y_reg_train, y_reg_test, X, y_reg, "REG")
    results["EXP7_Stacking"] = {"clf": m_clf7, "reg": m_reg7,
        "description": "Stacking: RF + GBR → Ridge/Logistic meta-learner"}

    return results

# =============================================================================
# TABLA RESUMEN Y FIGURAS COMPARATIVAS
# =============================================================================

def plot_comparison(results, out_dir):
    section("Tabla comparativa de experimentos")

    # Filtrar experimentos con resultados válidos
    valid = {k: v for k, v in results.items() if v["clf"] is not None}

    exp_names  = list(valid.keys())
    short_names = [k.replace("EXP", "E").replace("_", " ") for k in exp_names]

    # ---- Clasificación ----
    acc_test = [v["clf"]["accuracy_test"]    for v in valid.values()]
    f1_macro = [v["clf"]["f1_macro"]         for v in valid.values()]
    cv_acc   = [v["clf"]["cv_accuracy_mean"] for v in valid.values()]
    cv_acc_s = [v["clf"]["cv_accuracy_std"]  for v in valid.values()]

    # ---- Regresión ----
    r2_test  = [v["reg"]["r2_test"]       for v in valid.values()]
    mae_test = [v["reg"]["mae_test"]      for v in valid.values()]
    cv_r2    = [v["reg"]["cv_r2_mean"]    for v in valid.values()]
    cv_r2_s  = [v["reg"]["cv_r2_std"]     for v in valid.values()]

    x = np.arange(len(exp_names))
    w = 0.28

    # Figura 1 — Clasificación
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(x - w/2, acc_test, w, label="Acc test",   color="#378ADD", alpha=0.85)
    axes[0].bar(x + w/2, cv_acc,   w, label="CV Acc",     color="#1D9E75", alpha=0.85,
                yerr=cv_acc_s, capsize=4)
    axes[0].set_xticks(x); axes[0].set_xticklabels(short_names, rotation=30, ha="right")
    axes[0].set_ylim(0, 1.05); axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Clasificación (atomtype) — comparación")
    axes[0].legend(); axes[0].axhline(acc_test[0], color="gray", linestyle=":", alpha=0.6)

    axes[1].bar(x, f1_macro, color="#534AB7", alpha=0.85)
    axes[1].set_xticks(x); axes[1].set_xticklabels(short_names, rotation=30, ha="right")
    axes[1].set_ylim(0, 1.05); axes[1].set_ylabel("F1 macro")
    axes[1].set_title("F1 macro test — comparación")
    axes[1].axhline(f1_macro[0], color="gray", linestyle=":", alpha=0.6)

    plt.suptitle(f"Comparación de experimentos — Clasificación {LABEL}", y=1.02)
    plt.tight_layout()
    save(fig, out_dir, "fig_compare_clf.png")

    # Figura 2 — Regresión
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(x - w/2, r2_test, w, label="R² test",  color="#378ADD", alpha=0.85)
    axes[0].bar(x + w/2, cv_r2,   w, label="CV R²",    color="#1D9E75", alpha=0.85,
                yerr=cv_r2_s, capsize=4)
    axes[0].set_xticks(x); axes[0].set_xticklabels(short_names, rotation=30, ha="right")
    axes[0].set_ylabel("R²"); axes[0].set_title("Regresión (charge) — R²")
    axes[0].legend(); axes[0].axhline(r2_test[0], color="gray", linestyle=":", alpha=0.6)

    axes[1].bar(x, mae_test, color="#D85A30", alpha=0.85)
    axes[1].set_xticks(x); axes[1].set_xticklabels(short_names, rotation=30, ha="right")
    axes[1].set_ylabel("MAE (e)"); axes[1].set_title("MAE test — comparación (menor = mejor)")
    axes[1].axhline(mae_test[0], color="gray", linestyle=":", alpha=0.6)

    plt.suptitle(f"Comparación de experimentos — Regresión {LABEL}", y=1.02)
    plt.tight_layout()
    save(fig, out_dir, "fig_compare_reg.png")

    # Figura 3 — Scatter CV R² std (estabilidad)
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(cv_r2, cv_r2_s, s=120, zorder=5,
                         c=range(len(exp_names)), cmap="viridis")
    for i, name in enumerate(short_names):
        ax.annotate(name, (cv_r2[i], cv_r2_s[i]),
                    textcoords="offset points", xytext=(8, 4), fontsize=9)
    ax.set_xlabel("CV R² media (mayor = mejor)")
    ax.set_ylabel("CV R² std (menor = más estable)")
    ax.set_title(f"Estabilidad vs rendimiento — Regresión {LABEL}\n"
                 f"Ideal: esquina superior izquierda")
    ax.invert_yaxis()
    plt.tight_layout()
    save(fig, out_dir, "fig_compare_stability.png")

    # Tabla en consola
    print(f"\n{'─'*100}")
    print(f"{'Experimento':<30} {'Acc test':>9} {'CV Acc':>9} {'R² test':>9} "
          f"{'CV R²':>9} {'CV R² std':>10} {'MAE':>8}")
    print(f"{'─'*100}")
    for k, v in valid.items():
        name = k.replace("EXP", "E").replace("_", " ")
        print(f"  {name:<28} {v['clf']['accuracy_test']:>9.4f} "
              f"{v['clf']['cv_accuracy_mean']:>9.4f} "
              f"{v['reg']['r2_test']:>9.4f} "
              f"{v['reg']['cv_r2_mean']:>9.4f} "
              f"{v['reg']['cv_r2_std']:>10.4f} "
              f"{v['reg']['mae_test']:>8.4f}")
    print(f"{'─'*100}")
    print("  Línea gris punteada en figuras = nivel del baseline (EXP 1)")

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=f"Comparación de experimentos — {LABEL}"
    )
    parser.add_argument("--input",      required=True)
    parser.add_argument("--output_dir", default="results_compare/")
    args    = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    section(f"COMPARACIÓN DE EXPERIMENTOS — {LABEL}")
    print("  EXP 1: RF baseline (referencia)")
    print("  EXP 2: RF conservador (leaf=5, depth=None)")
    print("  EXP 3: RF con max_features=sqrt, n=500")
    print("  EXP 4: GradientBoosting")
    print("  EXP 5: XGBoost (si está instalado)")
    print(f"  EXP 6: RF + feature selection (top {TOP_FEATURES})")
    print("  EXP 7: Stacking RF+GBR → meta-learner")

    # Cargar datos
    (X, X_train, X_test,
     y_clf, y_clf_train, y_clf_test,
     y_reg, y_reg_train, y_reg_test,
     feature_cols) = load_and_split(Path(args.input))

    # Correr experimentos
    section("Corriendo experimentos")
    results = run_experiments(
        X, X_train, X_test,
        y_clf, y_clf_train, y_clf_test,
        y_reg, y_reg_train, y_reg_test,
        feature_cols
    )

    # Figuras y tabla comparativa
    plot_comparison(results, out_dir)

    # Guardar JSON con todos los resultados
    metrics_path = out_dir / "metrics_compare.json"
    # Limpiar resultados no serializables
    results_clean = {
        k: {kk: vv for kk, vv in v.items() if kk not in ["clf", "reg"]}
        for k, v in results.items()
    }
    with open(metrics_path, "w") as f:
        json.dump(results_clean, f, indent=2)
    print(f"\n  → {metrics_path.name}")

    section("LISTO")
    print(f"  Resultados en: {out_dir}")
    print(f"  Figuras:")
    print(f"    fig_compare_clf.png       → clasificación")
    print(f"    fig_compare_reg.png       → regresión")
    print(f"    fig_compare_stability.png → estabilidad CV vs rendimiento")
    print(f"    metrics_compare.json      → todas las métricas")

if __name__ == "__main__":
    main()
