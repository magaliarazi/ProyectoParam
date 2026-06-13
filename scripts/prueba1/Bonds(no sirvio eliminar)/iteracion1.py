import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
# =========================
# LOAD DATA
# =========================
df = pd.read_csv("bond_dataset.csv")

# =========================
# ENCODING CATEGORICAL
# =========================
le_mol = LabelEncoder()
le_i = LabelEncoder()
le_j = LabelEncoder()

df["molecule"] = le_mol.fit_transform(df["molecule"])
df["atom_i_type"] = le_i.fit_transform(df["atom_i_type"])
df["atom_j_type"] = le_j.fit_transform(df["atom_j_type"])

# =========================
# FEATURES / TARGET
# =========================
X = df[["molecule", "atom_i", "atom_j", "atom_i_type", "atom_j_type"]]
y = df["bond_exists"]

# =========================
# TRAIN / TEST
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# MODEL
# =========================
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print("Matriz de confusión:")
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

print(classification_report(y_test, y_pred))