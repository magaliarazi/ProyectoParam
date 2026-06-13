"""
model_AA_v2.py
--------------
Versión mejorada de model_AA.py con dos cambios clave:

  OPCIÓN 6 — Split por molécula completa (evita data leakage)
    El split original mezclaba átomos de la misma molécula en train y test,
    lo que infla las métricas. Ahora el split se hace por molécula: todas
    las instancias de una misma molécula van al mismo conjunto.

  OPCIÓN 4 — Cross-validation por molécula (GroupKFold)
    El CV original (StratifiedKFold aleatorio) provocaba std=0.26 en R²
    porque distintos folds veían distintas moléculas. GroupKFold garantiza
    que cada molécula aparece en un solo fold, dando una estimación real
    de cómo generaliza el modelo a moléculas nuevas.

REQUISITO: el dataset debe tener una columna que identifique la molécula
  de cada átomo. Por defecto se llama "mol_id". Si se llama distinto,
  cambiá la constante MOL_ID_COL abajo.
  Si el dataset NO tiene columna de molécula, el script puede inferirla
  automáticamente agrupando átomos consecutivos (ver --infer_mol_id).

Tareas:
  - Clasificación: predicción de atomtype
  - Regresión:     predicción de charge

Salidas:
  results_AA_v2/
    ├── clf_AA.joblib
    ├── reg_AA.joblib
    ├── metrics_AA.json
    ├── fig_mol_split_info.png          distribución train/test por molécula
    ├── fig_atomtype_distribution.png
    ├── fig_confusion_matrix.png
    ├── fig_feature_importance_clf.png
    ├── fig_feature_importance_reg.png
    ├── fig_charge_predicted_vs_real.png
    ├── fig_charge_residuals.png
    └── fig_crossval_scores.png

Uso:
    python model_AA_v2.py --input processed/aa_clean.csv --output_dir results_AA_v2/

    # Si la columna de molécula no se llama "mol_id":
    python model_AA_v2.py --input processed/aa_clean.csv --mol_col nombre_columna

    # Si no hay columna de molécula e inferir por grupos consecutivos:
    python model_AA_v2.py --input processed/aa_clean.csv --infer_mol_id --atoms_per_mol 14

Dependencias:
    pip install pandas numpy scikit-learn matplotlib seaborn joblib
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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.model_selection import (
    GroupKFold, cross_val_score, cross_validate,
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"]  = 11

LABEL = "AA"

# =============================================================================
# CONFIG
# =============================================================================

TARGET_CLF   = "atomtype"
TARGET_REG   = "charge"
TEST_SIZE    = 0.20        # fracción de moléculas para test
RANDOM_SEED  = 42
CV_FOLDS     = 5           # número de folds en GroupKFold

# Nombre de la columna que identifica la molécula de cada átomo
# Cambiá esto si tu columna se llama distinto
MOL_ID_COL   = "mol_id"

RF_CLF_PARAMS = dict(
    n_estimators     = 150,
    max_depth        = 10,
    min_samples_leaf = 1,
    class_weight     = "balanced",
    random_state     = RANDOM_SEED,
    n_jobs           = -1,
)

RF_REG_PARAMS = dict(
    n_estimators     = 150,
    max_depth        = 10,
    min_samples_leaf = 1,
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
# INFERENCIA DE mol_id (si no existe la columna)
# =============================================================================

def infer_mol_id(df, atoms_per_mol):
    """
    Crea una columna mol_id asumiendo que cada grupo de `atoms_per_mol`
    filas consecutivas pertenece a la misma molécula.
    Usá esto solo si el dataset no tiene columna de molécula.
    """
    mol_ids = np.arange(len(df)) // atoms_per_mol
    print(f"  Mol IDs inferidos: {mol_ids.max() + 1} moléculas "
          f"(~{atoms_per_mol} átomos cada una)")
    return mol_ids


# =============================================================================
# OPCIÓN 6 — SPLIT POR MOLÉCULA
# =============================================================================

def molecule_train_test_split(df, mol_col, test_size=0.20, random_state=42):
    """
    Divide el dataset en train y test a nivel de MOLÉCULA completa,
    no a nivel de átomo individual.

    Esto evita que átomos de la misma molécula aparezcan en train y test
    al mismo tiempo (data leakage estructural).

    Procedimiento:
      1. Obtener la lista de moléculas únicas
      2. Mezclarlas aleatoriamente
      3. Asignar el 80% de moléculas a train y 20% a test
      4. Filtrar el dataframe según la asignación

    Retorna: índices de train y test (sobre el dataframe original)
    """
    rng      = np.random.default_rng(random_state)
    mols     = df[mol_col].unique()
    n_mols   = len(mols)
    n_test   = max(1, int(np.ceil(n_mols * test_size)))
    n_train  = n_mols - n_test

    shuffled = rng.permutation(mols)
    train_mols = set(shuffled[:n_train])
    test_mols  = set(shuffled[n_train:])

    train_idx = df.index[df[mol_col].isin(train_mols)].tolist()
    test_idx  = df.index[df[mol_col].isin(test_mols)].tolist()

    print(f"\n  Split por molécula:")
    print(f"    Total moléculas:  {n_mols}")
    print(f"    Moléculas train:  {n_train}  ({n_train/n_mols*100:.0f}%)")
    print(f"    Moléculas test:   {n_test}   ({n_test/n_mols*100:.0f}%)")
    print(f"    Átomos train:     {len(train_idx)}")
    print(f"    Átomos test:      {len(test_idx)}")

    return train_idx, test_idx, sorted(train_mols), sorted(test_mols)


# =============================================================================
# CARGA Y SPLIT
# =============================================================================

def load_and_split(csv_path: Path, mol_col: str,
                   infer_mol: bool, atoms_per_mol: int):
    """
    Carga el dataset y aplica el split por molécula completa (Opción 6).
    """
    df = pd.read_csv(csv_path)
    print(f"Dataset cargado: {df.shape}")

    # --- Gestión de la columna mol_id ---
    if infer_mol:
        df[mol_col] = infer_mol_id(df, atoms_per_mol)
        print(f"  Columna '{mol_col}' creada por inferencia")
    elif mol_col not in df.columns:
        raise ValueError(
            f"No se encontró la columna '{mol_col}' en el dataset.\n"
            f"  Columnas disponibles: {list(df.columns)}\n"
            f"  Opciones:\n"
            f"    1) Pasá --mol_col <nombre> con el nombre correcto\n"
            f"    2) Usá --infer_mol_id --atoms_per_mol <N> para inferirla"
        )

    print(f"  Columna molécula:  '{mol_col}'")
    print(f"  Moléculas únicas:  {df[mol_col].nunique()}")
    print(f"  Clases atomtype:   {df[TARGET_CLF].nunique()}")
    print(f"  Rango charge:      [{df[TARGET_REG].min():.3f}, {df[TARGET_REG].max():.3f}]")

    # --- Features y targets ---
    drop_cols    = [TARGET_CLF, TARGET_REG, mol_col]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X      = df[feature_cols]
    y_clf  = df[TARGET_CLF]
    y_reg  = df[TARGET_REG]
    groups = df[mol_col]   # vector de grupos para GroupKFold

    # Eliminar clases con < 2 instancias
    MIN_SAMPLES = 2
    counts  = y_clf.value_counts()
    removed = sorted(counts[counts < MIN_SAMPLES].index.tolist())
    mask    = y_clf.isin(counts[counts >= MIN_SAMPLES].index)
    X       = X[mask].reset_index(drop=True)
    y_clf   = y_clf[mask].reset_index(drop=True)
    y_reg   = y_reg[mask].reset_index(drop=True)
    groups  = groups[mask].reset_index(drop=True)
    df_filt = df[mask].reset_index(drop=True)

    print(f"  Clases eliminadas (1 instancia): {removed}")
    print(f"  Átomos restantes:  {len(X)}")

    non_numeric = X.select_dtypes(include=["object"]).columns.tolist()
    if non_numeric:
        print(f"  ⚠ Columnas no numéricas: {non_numeric}")

    # --- OPCIÓN 6: split por molécula ---
    train_idx, test_idx, train_mols, test_mols = molecule_train_test_split(
        df_filt, mol_col, TEST_SIZE, RANDOM_SEED
    )

    X_train      = X.iloc[train_idx].reset_index(drop=True)
    X_test       = X.iloc[test_idx].reset_index(drop=True)
    y_clf_train  = y_clf.iloc[train_idx].reset_index(drop=True)
    y_clf_test   = y_clf.iloc[test_idx].reset_index(drop=True)
    y_reg_train  = y_reg.iloc[train_idx].reset_index(drop=True)
    y_reg_test   = y_reg.iloc[test_idx].reset_index(drop=True)
    groups_train = groups.iloc[train_idx].reset_index(drop=True)

    return (X, X_train, X_test,
            y_clf, y_clf_train, y_clf_test,
            y_reg, y_reg_train, y_reg_test,
            groups, groups_train,
            feature_cols, train_mols, test_mols)


# =============================================================================
# VISUALIZACIÓN — info del split por molécula
# =============================================================================

def plot_mol_split_info(df_orig, mol_col, train_mols, test_mols, out_dir):
    """
    Muestra cuántos átomos hay por molécula y cuáles fueron a train vs test.
    """
    section("Distribución del split por molécula")

    counts = df_orig.groupby(mol_col).size().reset_index(name="n_atoms")
    counts["split"] = counts[mol_col].apply(
        lambda m: "Train" if m in set(train_mols) else "Test"
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Barras por molécula
    colors = {"Train": "#378ADD", "Test": "#D85A30"}
    for split, grp in counts.groupby("split"):
        axes[0].bar(grp[mol_col].astype(str), grp["n_atoms"],
                    label=split, color=colors[split], alpha=0.85)
    axes[0].set_xlabel("Molécula")
    axes[0].set_ylabel("Número de átomos")
    axes[0].set_title(f"Átomos por molécula — {LABEL}")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].legend()

    # Pie chart
    summary = counts.groupby("split")["n_atoms"].sum()
    axes[1].pie(summary.values, labels=summary.index,
                colors=[colors[s] for s in summary.index],
                autopct="%1.1f%%", startangle=90)
    axes[1].set_title(f"Proporción átomos train/test\n(split por molécula completa)")

    plt.suptitle(f"Split por molécula — {LABEL}", y=1.02)
    plt.tight_layout()
    save(fig, out_dir, "fig_mol_split_info.png")


# =============================================================================
# VISUALIZACIÓN — distribución de clases
# =============================================================================

def plot_class_distribution(y_clf, y_clf_train, y_clf_test, out_dir):
    section("Distribución de clases")

    counts_full  = y_clf.value_counts()
    counts_train = y_clf_train.value_counts()
    counts_test  = y_clf_test.value_counts()

    all_classes  = counts_full.index
    counts_train = counts_train.reindex(all_classes, fill_value=0)
    counts_test  = counts_test.reindex(all_classes, fill_value=0)

    x = np.arange(len(all_classes))
    w = 0.35

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - w/2, counts_train.values, w, label="Train", color="#378ADD")
    ax.bar(x + w/2, counts_test.values,  w, label="Test",  color="#D85A30")
    ax.set_xticks(x)
    ax.set_xticklabels(all_classes, rotation=45, ha="right")
    ax.set_xlabel("atomtype")
    ax.set_ylabel("Frecuencia")
    ax.set_title(f"Distribución de clases — {LABEL} (Train vs Test, split por molécula)")
    ax.legend()
    plt.tight_layout()
    save(fig, out_dir, "fig_atomtype_distribution.png")


# =============================================================================
# CLASIFICACIÓN
# =============================================================================

def train_classifier(X_train, y_train):
    section("Entrenando clasificador (atomtype)")
    clf = RandomForestClassifier(**RF_CLF_PARAMS)
    clf.fit(X_train, y_train)
    print("  ✓ Clasificador entrenado")
    return clf


def evaluate_classifier(clf, X_train, X_test, y_train, y_test,
                         X_full, y_full, groups_full, feature_cols, out_dir):
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

    # ------------------------------------------------------------------
    # OPCIÓN 4: Cross-validation por molécula con GroupKFold
    # ------------------------------------------------------------------
    # GroupKFold garantiza que todos los átomos de una misma molécula
    # queden en el mismo fold, evitando la filtración de información
    # entre train y validation que inflaba/deflaba el CV anterior.
    # n_splits = mín(CV_FOLDS, n_moléculas) para evitar errores.
    # ------------------------------------------------------------------
    section("Cross-validation por molécula — Clasificación (GroupKFold)")
    n_mols_cv = groups_full.nunique()
    n_splits_clf = min(CV_FOLDS, n_mols_cv)
    if n_splits_clf < CV_FOLDS:
        print(f"  ⚠ Solo {n_mols_cv} moléculas disponibles → usando {n_splits_clf} folds")

    gkf = GroupKFold(n_splits=n_splits_clf)

    cv_results = cross_validate(
        RandomForestClassifier(**RF_CLF_PARAMS),
        X_full, y_full,
        groups   = groups_full,
        cv       = gkf,
        scoring  = ["accuracy", "f1_macro", "f1_weighted"],
        return_train_score = True,
        n_jobs   = -1,
    )
    cv_acc = cv_results["test_accuracy"]
    cv_f1m = cv_results["test_f1_macro"]
    cv_f1w = cv_results["test_f1_weighted"]

    print(f"  CV Accuracy:    {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
    print(f"  CV F1 macro:    {cv_f1m.mean():.4f} ± {cv_f1m.std():.4f}")
    print(f"  CV F1 weighted: {cv_f1w.mean():.4f} ± {cv_f1w.std():.4f}")
    print(f"  (GroupKFold, {n_splits_clf} folds, {n_mols_cv} moléculas)")

    # Figura — Matriz de confusión
    classes = sorted(y_full.unique())
    cm      = confusion_matrix(y_test, y_pred_test, labels=classes)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes,
                linewidths=0.5, ax=ax)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusión — {LABEL} (test set, split por molécula)\n"
                 f"(valores: conteo, color: proporción por fila)")
    plt.tight_layout()
    save(fig, out_dir, "fig_confusion_matrix.png")

    plot_feature_importance(clf, feature_cols, out_dir,
                            "fig_feature_importance_clf.png",
                            f"Feature importance — Clasificación {LABEL}")
    plot_crossval(cv_acc, cv_f1m, cv_f1w, n_splits_clf, out_dir)

    return {
        "accuracy_train":  round(acc_train, 4),
        "accuracy_test":   round(acc_test,  4),
        "f1_macro":        round(f1_macro,  4),
        "f1_weighted":     round(f1_w,      4),
        "cv_type":         "GroupKFold por molécula",
        "cv_folds_used":   n_splits_clf,
        "cv_accuracy_mean": round(float(cv_acc.mean()), 4),
        "cv_accuracy_std":  round(float(cv_acc.std()),  4),
        "cv_f1_macro_mean": round(float(cv_f1m.mean()), 4),
        "cv_f1_macro_std":  round(float(cv_f1m.std()),  4),
        "cv_f1_weighted_mean": round(float(cv_f1w.mean()), 4),
        "cv_f1_weighted_std":  round(float(cv_f1w.std()),  4),
        "classification_report": classification_report(
            y_test, y_pred_test, zero_division=0, output_dict=True
        ),
    }


# =============================================================================
# REGRESIÓN
# =============================================================================

def train_regressor(X_train, y_train):
    section("Entrenando regresor (charge)")
    reg = RandomForestRegressor(**RF_REG_PARAMS)
    reg.fit(X_train, y_train)
    print("  ✓ Regresor entrenado")
    return reg


def evaluate_regressor(reg, X_train, X_test, y_train, y_test,
                        X_full, y_full, groups_full, feature_cols, out_dir):
    section("Evaluación — Regresión")

    y_pred_train = reg.predict(X_train)
    y_pred_test  = reg.predict(X_test)

    mae_train  = mean_absolute_error(y_train, y_pred_train)
    mae_test   = mean_absolute_error(y_test,  y_pred_test)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test  = np.sqrt(mean_squared_error(y_test,  y_pred_test))
    r2_train   = r2_score(y_train, y_pred_train)
    r2_test    = r2_score(y_test,  y_pred_test)

    print(f"  MAE  train: {mae_train:.4f} e  |  test: {mae_test:.4f} e")
    print(f"  RMSE train: {rmse_train:.4f} e  |  test: {rmse_test:.4f} e")
    print(f"  R²   train: {r2_train:.4f}    |  test: {r2_test:.4f}")

    # ------------------------------------------------------------------
    # OPCIÓN 4: Cross-validation por molécula con GroupKFold
    # ------------------------------------------------------------------
    section("Cross-validation por molécula — Regresión (GroupKFold)")
    n_mols_cv    = groups_full.nunique()
    n_splits_reg = min(CV_FOLDS, n_mols_cv)
    if n_splits_reg < CV_FOLDS:
        print(f"  ⚠ Solo {n_mols_cv} moléculas disponibles → usando {n_splits_reg} folds")

    gkf = GroupKFold(n_splits=n_splits_reg)

    cv_mae = -cross_val_score(
        RandomForestRegressor(**RF_REG_PARAMS), X_full, y_full,
        groups=groups_full, cv=gkf,
        scoring="neg_mean_absolute_error", n_jobs=-1
    )
    cv_rmse = np.sqrt(-cross_val_score(
        RandomForestRegressor(**RF_REG_PARAMS), X_full, y_full,
        groups=groups_full, cv=gkf,
        scoring="neg_mean_squared_error", n_jobs=-1
    ))
    cv_r2 = cross_val_score(
        RandomForestRegressor(**RF_REG_PARAMS), X_full, y_full,
        groups=groups_full, cv=gkf,
        scoring="r2", n_jobs=-1
    )

    print(f"  CV MAE:  {cv_mae.mean():.4f} ± {cv_mae.std():.4f} e")
    print(f"  CV RMSE: {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f} e")
    print(f"  CV R²:   {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
    print(f"  (GroupKFold, {n_splits_reg} folds, {n_mols_cv} moléculas)")

    # Figura — Predicho vs Real
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, y_pred_test, alpha=0.6, color="#378ADD",
               edgecolors="white", linewidth=0.3, s=50, label="Test")
    lims = [min(y_test.min(), y_pred_test.min()) - 0.05,
            max(y_test.max(), y_pred_test.max()) + 0.05]
    ax.plot(lims, lims, "r--", linewidth=1.2, label="Predicción perfecta")
    ax.set_xlabel("Charge real (e)")
    ax.set_ylabel("Charge predicha (e)")
    ax.set_title(f"Charge real vs predicha — {LABEL}\n"
                 f"MAE={mae_test:.4f} | RMSE={rmse_test:.4f} | R²={r2_test:.4f}\n"
                 f"(test = moléculas no vistas en train)")
    ax.legend()
    plt.tight_layout()
    save(fig, out_dir, "fig_charge_predicted_vs_real.png")

    # Figura — Residuos
    residuals = y_test - y_pred_test
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].scatter(y_pred_test, residuals, alpha=0.6,
                    color="#534AB7", edgecolors="white", linewidth=0.3, s=50)
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Charge predicha (e)")
    axes[0].set_ylabel("Residuo (real - predicho)")
    axes[0].set_title("Residuos vs Charge predicha")

    axes[1].hist(residuals, bins=25, color="#534AB7", edgecolor="white")
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1)
    axes[1].axvline(residuals.mean(), color="orange", linestyle="--",
                    linewidth=1, label=f"Media: {residuals.mean():.4f}")
    axes[1].set_xlabel("Residuo (e)")
    axes[1].set_ylabel("Frecuencia")
    axes[1].set_title("Distribución de residuos")
    axes[1].legend()

    plt.suptitle(f"Análisis de residuos — {LABEL}", y=1.02)
    plt.tight_layout()
    save(fig, out_dir, "fig_charge_residuals.png")

    plot_feature_importance(reg, feature_cols, out_dir,
                            "fig_feature_importance_reg.png",
                            f"Feature importance — Regresión {LABEL}")

    return {
        "mae_train":   round(mae_train,  4),
        "mae_test":    round(mae_test,   4),
        "rmse_train":  round(rmse_train, 4),
        "rmse_test":   round(rmse_test,  4),
        "r2_train":    round(r2_train,   4),
        "r2_test":     round(r2_test,    4),
        "cv_type":     "GroupKFold por molécula",
        "cv_folds_used": n_splits_reg,
        "cv_mae_mean":  round(float(cv_mae.mean()),  4),
        "cv_mae_std":   round(float(cv_mae.std()),   4),
        "cv_rmse_mean": round(float(cv_rmse.mean()), 4),
        "cv_rmse_std":  round(float(cv_rmse.std()),  4),
        "cv_r2_mean":   round(float(cv_r2.mean()),   4),
        "cv_r2_std":    round(float(cv_r2.std()),    4),
    }


# =============================================================================
# FIGURAS COMPARTIDAS
# =============================================================================

def plot_feature_importance(model, feature_cols, out_dir, filename, title):
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    top = importances.nlargest(TOP_N_FEATURES).sort_values()

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = sns.color_palette("muted", len(top))
    ax.barh(top.index, top.values, color=colors)
    ax.set_xlabel("Importancia (Mean Decrease Impurity)")
    ax.set_title(f"{title}\n(top {TOP_N_FEATURES} features)")
    plt.tight_layout()
    save(fig, out_dir, filename)


def plot_crossval(cv_acc, cv_f1m, cv_f1w, n_splits, out_dir):
    folds = np.arange(1, n_splits + 1)
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
        ax.set_xlabel("Fold (molécula)")
        ax.set_ylabel(title)
        ax.set_title(f"{title} por fold")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=9)

    plt.suptitle(f"Cross-validation GroupKFold ({n_splits} folds, por molécula) — {LABEL}",
                 y=1.02)
    plt.tight_layout()
    save(fig, out_dir, "fig_crossval_scores.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=f"Modelo Random Forest — {LABEL} (split + CV por molécula)"
    )
    parser.add_argument("--input",        required=True,
                        help="CSV limpio (aa_clean.csv)")
    parser.add_argument("--output_dir",   default=f"results_{LABEL}_v2/")
    parser.add_argument("--mol_col",      default=MOL_ID_COL,
                        help=f"Nombre de la columna de molécula (default: '{MOL_ID_COL}')")
    parser.add_argument("--infer_mol_id", action="store_true",
                        help="Inferir mol_id agrupando filas consecutivas")
    parser.add_argument("--atoms_per_mol", type=int, default=14,
                        help="Átomos por molécula para inferencia (default: 14)")
    args    = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  MODELO RANDOM FOREST — {LABEL} v2")
    print(f"  Split por molécula + GroupKFold CV")
    print(f"{'='*55}")

    # Cargar y dividir por molécula
    (X, X_train, X_test,
     y_clf, y_clf_train, y_clf_test,
     y_reg, y_reg_train, y_reg_test,
     groups, groups_train,
     feature_cols,
     train_mols, test_mols) = load_and_split(
        Path(args.input), args.mol_col,
        args.infer_mol_id, args.atoms_per_mol
    )

    # Cargar df original para la figura de split (necesita mol_col)
    df_orig = pd.read_csv(args.input)
    if args.infer_mol_id:
        df_orig[args.mol_col] = infer_mol_id(df_orig, args.atoms_per_mol)

    plot_mol_split_info(df_orig, args.mol_col, train_mols, test_mols, out_dir)
    plot_class_distribution(y_clf, y_clf_train, y_clf_test, out_dir)

    # Clasificación
    clf         = train_classifier(X_train, y_clf_train)
    metrics_clf = evaluate_classifier(
        clf, X_train, X_test, y_clf_train, y_clf_test,
        X, y_clf, groups, feature_cols, out_dir
    )

    # Regresión
    reg         = train_regressor(X_train, y_reg_train)
    metrics_reg = evaluate_regressor(
        reg, X_train, X_test, y_reg_train, y_reg_test,
        X, y_reg, groups, feature_cols, out_dir
    )

    # Guardar modelos
    section("Guardando modelos")
    clf_path = out_dir / f"clf_{LABEL}.joblib"
    reg_path = out_dir / f"reg_{LABEL}.joblib"
    dump(clf, clf_path)
    dump(reg, reg_path)
    print(f"  → {clf_path.name}")
    print(f"  → {reg_path.name}")

    # Guardar métricas
    metrics = {
        "dataset":        LABEL,
        "version":        "v2 — split y CV por molécula",
        "mol_col":        args.mol_col,
        "n_molecules":    int(groups.nunique()),
        "n_train":        len(X_train),
        "n_test":         len(X_test),
        "n_features":     len(feature_cols),
        "test_size":      TEST_SIZE,
        "cv_folds":       CV_FOLDS,
        "train_mols":     [str(m) for m in train_mols],
        "test_mols":      [str(m) for m in test_mols],
        "rf_clf_params":  RF_CLF_PARAMS,
        "rf_reg_params":  RF_REG_PARAMS,
        "classification": metrics_clf,
        "regression":     metrics_reg,
    }
    metrics_path = out_dir / f"metrics_{LABEL}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  → {metrics_path.name}")

    # Resumen final
    section("RESUMEN FINAL")
    print(f"  Split: {len(set(train_mols))} moléculas train / "
          f"{len(set(test_mols))} moléculas test")
    print(f"\n  Clasificación (atomtype):")
    print(f"    Accuracy test:   {metrics_clf['accuracy_test']:.4f}")
    print(f"    F1 macro test:   {metrics_clf['f1_macro']:.4f}")
    print(f"    CV Accuracy:     {metrics_clf['cv_accuracy_mean']:.4f} "
          f"± {metrics_clf['cv_accuracy_std']:.4f}  [GroupKFold]")
    print(f"\n  Regresión (charge):")
    print(f"    MAE test:        {metrics_reg['mae_test']:.4f} e")
    print(f"    RMSE test:       {metrics_reg['rmse_test']:.4f} e")
    print(f"    R² test:         {metrics_reg['r2_test']:.4f}")
    print(f"    CV R²:           {metrics_reg['cv_r2_mean']:.4f} "
          f"± {metrics_reg['cv_r2_std']:.4f}  [GroupKFold]")

    print(f"\n✔ Resultados guardados en: {out_dir}")


if __name__ == "__main__":
    main()
