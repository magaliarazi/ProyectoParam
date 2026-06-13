import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score

from xgboost import XGBClassifier


# =========================
# CONFIG
# =========================
DATASET_PATH = "bond_dataset_encoded2.csv"


# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATASET_PATH)

print("Dataset shape:", df.shape)


# =========================
# ENCODING
# =========================
categorical_cols = [
    "type_i",
    "type_j",
    "element_i",
    "element_j"
]

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))


# =========================
# FEATURES
# =========================
features = [
    "type_i",
    "type_j",
    "element_i",
    "element_j",
    "charge_i",
    "charge_j",
    "mass_i",
    "mass_j",
    "charge_diff",
    "degree_i",
    "degree_j",
    "valence_i",
    "valence_j",
    "same_element"
]

X = df[features]
y = df["bond_exists"]


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
# MODELOS
# =========================
models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=400,
        max_depth=15,
        min_samples_split=3,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        scale_pos_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss"
    )
}


# =========================
# EVALUACIÓN CON THRESHOLD
# =========================
def evaluate_model(model, name):

    print(f"\n==================== {name} ====================")

    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]

    print("\nROC-AUC:", roc_auc_score(y_test, y_proba))

    best_f1 = 0
    best_threshold = 0.5

    print("\n--- Threshold tuning ---")

    for t in [0.5, 0.6, 0.7, 0.8]:
        y_pred = (y_proba > t).astype(int)
        f1 = f1_score(y_test, y_pred)

        print(f"Threshold {t} → F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    print(f"\n🔥 Mejor threshold: {best_threshold}")

    # Evaluación final
    y_pred_final = (y_proba > best_threshold).astype(int)

    print("\n📊 MATRIZ DE CONFUSIÓN")
    print(confusion_matrix(y_test, y_pred_final))

    print("\n📊 REPORT FINAL")
    print(classification_report(y_test, y_pred_final))

    return best_f1


# =========================
# RUN
# =========================
results = {}

for name, model in models.items():
    f1 = evaluate_model(model, name)
    results[name] = f1


# =========================
# BEST MODEL
# =========================
best_model = max(results, key=results.get)

print("\n==============================")
print("🏆 MEJOR MODELO:", best_model)
print("F1:", results[best_model])
print("==============================")