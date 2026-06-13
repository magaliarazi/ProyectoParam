import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

# -----------------------------
# CONFIG
# -----------------------------
DATA_PATH = "angles_dataset_encoded.csv"
BATCH_SIZE = 128
EPOCHS = 20
LR = 1e-3
EMB_DIM = 8

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(DATA_PATH)

# -----------------------------
# PREPROCESS (ACA eliminamos columnas)
# -----------------------------
cols_to_drop = ["ai","aj","ak","element_i","element_j","element_k"]
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# targets
y = df["target_angle"].values.astype(np.float32)

# grupos (para evitar leakage)
groups = df["molecule"].values

# features
X = df.drop(columns=["target_angle", "target_fc", "molecule"])

# separar tipos vs numéricos
type_cols = ["type_i", "type_j", "type_k"]

X_types = X[type_cols].values.astype(np.int64)
X_numeric = X.drop(columns=type_cols).values.astype(np.float32)

# -----------------------------
# SPLIT POR MOLECULA
# -----------------------------
gss = GroupShuffleSplit(test_size=0.2, random_state=42)

train_idx, test_idx = next(gss.split(X_numeric, y, groups))

X_types_train = X_types[train_idx]
X_types_test = X_types[test_idx]

X_num_train = X_numeric[train_idx]
X_num_test = X_numeric[test_idx]

y_train = y[train_idx]
y_test = y[test_idx]

# -----------------------------
# NORMALIZAR NUMERICOS
# -----------------------------
scaler = StandardScaler()
X_num_train = scaler.fit_transform(X_num_train)
X_num_test = scaler.transform(X_num_test)

# -----------------------------
# TORCH DATA
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_tensor(x, dtype):
    return torch.tensor(x, dtype=dtype).to(device)

X_types_train = to_tensor(X_types_train, torch.long)
X_types_test = to_tensor(X_types_test, torch.long)

X_num_train = to_tensor(X_num_train, torch.float32)
X_num_test = to_tensor(X_num_test, torch.float32)

y_train = to_tensor(y_train, torch.float32).view(-1,1)
y_test = to_tensor(y_test, torch.float32).view(-1,1)

# -----------------------------
# MODEL
# -----------------------------
num_types = int(df[["type_i","type_j","type_k"]].max().max()) + 1
num_numeric = X_num_train.shape[1]

class AngleModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(num_types, EMB_DIM)

        self.net = nn.Sequential(
            nn.Linear(EMB_DIM*3 + num_numeric, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, types, numeric):
        ti = self.embedding(types[:,0])
        tj = self.embedding(types[:,1])
        tk = self.embedding(types[:,2])

        x = torch.cat([ti, tj, tk, numeric], dim=1)
        return self.net(x)

model = AngleModel().to(device)

# -----------------------------
# TRAIN
# -----------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

def train():
    model.train()
    optimizer.zero_grad()

    pred = model(X_types_train, X_num_train)
    loss = criterion(pred, y_train)

    loss.backward()
    optimizer.step()

    return loss.item()

def evaluate():
    model.eval()
    with torch.no_grad():
        pred = model(X_types_test, X_num_test)
        loss = criterion(pred, y_test)

        mae = torch.mean(torch.abs(pred - y_test)).item()
    return loss.item(), mae

# -----------------------------
# LOOP
# -----------------------------
for epoch in range(EPOCHS):
    train_loss = train()
    val_loss, val_mae = evaluate()

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | MAE: {val_mae:.4f}")

# -----------------------------
# FINAL
# -----------------------------
val_loss, val_mae = evaluate()
print("\n✅ RESULTADOS FINALES")
print(f"Test MSE: {val_loss:.4f}")
print(f"Test MAE: {val_mae:.4f}")