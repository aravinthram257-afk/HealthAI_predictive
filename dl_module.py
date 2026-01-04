import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from pathlib import Path
from data_loader import load_los
class LosDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
class LosNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
    def forward(self, x):
        return self.model(x)
def train_los_pytorch(epochs=10, batch_size=256, lr=1e-3):
    # ===== Load and prepare data =====
    df = load_los()
    target_col = "lengthofstay"
    y = df[target_col]
    X = (
        df.select_dtypes(include=["int64", "float64"])
        .drop(columns=[target_col], errors="ignore")
        .fillna(0)
    )
    y = y.fillna(y.median())
    print(f"DL – Dataset shape: {df.shape}")
    print(f"DL – Features: {X.shape[1]}, Target range: {y.min():.1f}-{y.max():.1f} days")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    # ===== Dataset & DataLoader =====
    train_ds = LosDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    # ===== Model, loss, optimizer =====
    model = LosNet(X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # ===== Training loop =====
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch).squeeze()
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(X_batch)
        epoch_loss /= len(train_ds)
        print(f"PyTorch Epoch {epoch+1}/{epochs}: Loss {epoch_loss:.4f}")
    # ===== Metrics on test set =====
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)
        preds = model(X_test_tensor).squeeze()
    mae = mean_absolute_error(y_test_tensor.numpy(), preds.numpy())
    rmse = np.sqrt(mean_squared_error(y_test_tensor.numpy(), preds.numpy()))
    r2 = r2_score(y_test_tensor.numpy(), preds.numpy())
    print(f"PyTorch LOS - MAE : {mae:.2f} days")
    print(f"PyTorch LOS - RMSE: {rmse:.2f} days")
    print(f"PyTorch LOS - R²  : {r2:.3f}")
    # ===== Save model after metrics =====
    base_dir = Path(__file__).resolve().parent.parent  # D:\healthai
    models_dir = base_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)      # create if missing
    model_path = models_dir / "los_net.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Saved PyTorch LOS model to {model_path}")
    print("PyTorch Deep Learning module ✅ COMPLETE")
    return model
if __name__ == "__main__":
    train_los_pytorch()
def load_los_model():
    base_dir = Path(__file__).resolve().parent.parent
    models_dir = base_dir / "models"
    model_path = models_dir / "los_net.pth"

    df = load_los()
    target_col = "lengthofstay"
    y = df[target_col]
    X = (
        df.select_dtypes(include=["int64", "float64"])
        .drop(columns=[target_col], errors="ignore")
        .fillna(0)
    )
    scaler = StandardScaler().fit(X)
    model = LosNet(X.shape[1])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model, scaler
