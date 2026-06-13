import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


# =========================
# CONFIG
# =========================
DATASET_PATH = "bond_dataset_encoded2.csv"


# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATASET_PATH)

print("Dataset shape:", df.shape)
print(df.head())


# =========================
# ENCODING
# =========================
categorical_cols = [
    "type_i",
    "type_j",
    "element_i",
    "element_j"
]

encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le


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
# MODEL
# =========================
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=15,
    min_samples_split=3,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


# =========================
# PREDICT
# =========================
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]


# =========================
# METRICS
# =========================
print("\n📊 MATRIZ DE CONFUSIÓN")
print(confusion_matrix(y_test, y_pred))

print("\n📊 REPORT")
print(classification_report(y_test, y_pred))

print("\n📊 ROC-AUC:", roc_auc_score(y_test, y_proba))


# =========================
# FEATURE IMPORTANCE
# =========================
importances = pd.Series(model.feature_importances_, index=features)
print("\n🔥 Feature importance:")
print(importances.sort_values(ascending=False))