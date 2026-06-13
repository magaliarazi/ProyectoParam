import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# =========================
# LOAD
# =========================
df = pd.read_csv("bond_dataset_encoded2.csv")

# =========================
# ENCODING
# =========================
cat_cols = ["element_i", "element_j", "type_i", "type_j"]

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))


# =========================
# FEATURES
# =========================
features = [
    "element_i", "type_i", "charge_i", "mass_i", "coord_i",
    "element_j", "type_j", "charge_j", "mass_j", "coord_j",
    "delta_charge", "delta_mass", "funct"
]

X = df[features]

# TARGETS
y = df[["target_c0", "target_c1"]]

# =========================
# SCALE TARGETS 🔥
# =========================
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)


# =========================
# SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_scaled, test_size=0.2, random_state=42
)


# =========================
# MODEL
# =========================
model = MultiOutputRegressor(
    RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )
)

model.fit(X_train, y_train)


# =========================
# PREDICT
# =========================
y_pred_scaled = model.predict(X_test)

# volver a escala original
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_real = scaler_y.inverse_transform(y_test)


# =========================
# METRICS
# =========================
print("\n📊 c0 MAE:", mean_absolute_error(y_test_real[:,0], y_pred[:,0]))
print("📊 c0 R2:", r2_score(y_test_real[:,0], y_pred[:,0]))

print("\n📊 c1 MAE:", mean_absolute_error(y_test_real[:,1], y_pred[:,1]))
print("📊 c1 R2:", r2_score(y_test_real[:,1], y_pred[:,1]))