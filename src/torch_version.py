import pandas as pd
import numpy as np

import mlflow.pytorch
import torch

from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)

mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment("torch_experiment")

new_project_root = str(Path(__file__).parent.parent / "data" / "salary_prediction_data_redacted")
print(new_project_root)

numeric_features = ["Experience", "Age"]
categorical_features = ["Education", "Location", "Job_Title", "Gender"]

df = pd.read_csv(new_project_root)
print("CSV прочитан:", df.shape, flush=True)
X = df.drop(columns=["Salary"])
y = df["Salary"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, # что делим
    test_size=0.4, # в X_temp, y_temp попадёт 40%
    random_state=42,
    # stratify Это нужно для сохранения пропорций классов при классификации. Только при классификации
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    random_state=42,
)

scaler = StandardScaler()

y_train_scaled = scaler.fit_transform(np.asarray(y_train).reshape(-1,1))
y_val_scaled = scaler.transform(np.asarray(y_val).reshape(-1,1))
y_test_scaled = scaler.transform(np.asarray(y_test).reshape(-1,1))

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore",
        sparse_output=False), categorical_features),
])

X_train = preprocessor.fit_transform(X_train).astype(np.float32)
X_val = preprocessor.fit_transform(X_val).astype(np.float32)
X_test = preprocessor.fit_transform(X_test).astype(np.float32)

def to_tensors(X, y):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(np.asarray(y).reshape(-1,1), dtype=torch.float32) # torch.long в многоклассовую классификацию, torch.float32 для регрессии

    return TensorDataset(X_tensor, y_tensor)

train_data = to_tensors(X_train, y_train_scaled)
val_data = to_tensors(X_val, y_val_scaled)
test_data = to_tensors(X_test, y_test_scaled)

batch_size = 32
epochs = 10

train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
)
val_loader = DataLoader(
    val_data,
    batch_size=batch_size,
)
test_loader = DataLoader(
    test_data,
    batch_size=batch_size,
)

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
        )
    def forward(self, x):
        return self.network(x)

model = NeuralNetwork(input_size=X_train.shape[1], output_size=1).to(device)
# многоклассовая классификация y.nunique() регрессия output_size=1

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train():
    model.train()
    total_loss = 0
    total = 0

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)

        prediction = model(X)
        loss = loss_fn(prediction, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        total += y.size(0)

    return total_loss / total

def evaluate(loader):
    model.eval()
    total_loss = 0
    total = 0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)

            prediction = model(X)
            loss = loss_fn(prediction, y)

            total_loss += loss.item() * X.size(0)
            total += y.size(0)

    return total_loss / total

best_val_loss = float("inf")

with mlflow.start_run(run_name="name"):

    mlflow.log_params({
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": 0.001,
        "optimizer": optimizer,
        "loss_fn": loss_fn,
        "input_size": X_train.shape[1],
        "output_size": 1,
        "device": str(device),
    })

    for epoch in range(epochs):
        train_loss = train()
        val_loss = evaluate(val_loader)

        print(f"train loss: {train_loss:.4f}")
        print(f"val loss: {val_loss:.4f}")

        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss,
        }, step=epoch + 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(model.state_dict(), "mlmodels/model.pth")

        model.load_state_dict(
            torch.load("mlmodels/model.pth", map_location=device, weights_only=True)
        )

    test_loss = evaluate(test_loader)
    test_rmse = np.sqrt(test_loss)
    print(f"Test RMSE: {test_rmse:.4f}")

    mlflow.log_metrics({
        "test_loss": test_loss,
        "test_rmse": test_rmse,
    })

    mlflow.log_artifact("mlmodels/model.pth")

    print(test_loss, test_rmse)