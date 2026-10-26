import pandas as pd
import numpy as np

import mlflow.pytorch
import torch
from sklearn.preprocessing import StandardScaler

from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)


mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("torch_experiment")

df = pd.read_csv('dataset.csv')

X = df.drop(columns=["target"])
y = df["target"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, # что делим
    test_size=0.4, # в X_temp, y_temp попадёт 40%
    random_state=42,
    stratify=y # только при классификации
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

def to_tensors(X, y):
    X_tensor = torch.tensor(X.to_numpy(dtype=np.float32),
                            dtype=torch.float32)
    y_tensor = torch.tensor(y.to_numpy(),
                            dtype=torch.long) # torch.long в многоклассовую классификацию, torch.float32 для регрессии

    return TensorDataset(X_tensor, y_tensor)

train_data = to_tensors(X_train, y_train)
val_data = to_tensors(X_val, y_val)
test_data = to_tensors(X_test, y_test)

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

model = NeuralNetwork(input_size=X_train.shape[1], output_size=y.nunique()).to(device)
# многоклассовая классификация y.nunique() регрессия output_size=1

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train():
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)

        prediction = model(X)
        loss = loss_fn(prediction, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        correct += (prediction.argmax(1) == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total

def evaluate(loader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)

            prediction = model(X)
            loss = loss_fn(prediction, y)

            total_loss += loss.item() * X.size(0)
            correct += (prediction.argmax(1) ==y).sum().item()
            total += y.size(0)

    return total_loss / total, correct / total

best_val_loss = float("inf")

with mlflow.start_run(run_name="name"):

    mlflow.log_params({
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": 0.001,
        "optimizer": optimizer,
        "loss_fn": loss_fn,
        "input_size": X_train.shape[1],
        "output_size": y.nunique(),
        "device": str(device),
    })

    for epoch in range(epochs):
        train_loss, train_acc = train()
        val_loss, val_acc = evaluate(val_loader)

        print(f"train loss: {train_loss:.4f}")
        print(f"train acc: {train_acc:.2%}")
        print(f"val loss: {val_loss:.4f}")
        print(f"val acc: {val_acc:.2%}")

        mlflow.log_metrics({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }, step=epoch + 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(model.state_dict(), "model.pth")

    model.load_state_dict(torch.load("model.pth"))

    test_loss, test_acc = evaluate(test_loader)

    mlflow.log_metrics({
        "test_loss": test_loss,
        "test_acc": test_acc,
    })

    mlflow.log_artifact("model.pth")

    print(test_loss, test_acc)