
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from lstm_model import TransitLSTM

BASE = Path("services/dl-service/data/processed")
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

BATCH_SIZE = 32
EPOCHS = 20


def load_data(split):
    data = np.load(BASE / f"{split}.npz")
    X, y = data["X"], data["y"]

    mask = np.any(X != 0, axis=(1, 2)) & (y != 0)

    X = torch.tensor(X[mask], dtype=torch.float32)
    y = torch.tensor(y[mask] - 1, dtype=torch.long)

    return TensorDataset(X, y)


train_loader = DataLoader(
    load_data("train"), batch_size=BATCH_SIZE, shuffle=True
)
val_loader = DataLoader(
    load_data("val"), batch_size=BATCH_SIZE
)
test_loader = DataLoader(
    load_data("test"), batch_size=BATCH_SIZE
)

model = TransitLSTM(num_classes=2).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(), lr=0.0003, weight_decay=0.0001
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2
)

save_path = Path("services/dl-service/models/transit_lstm.pth")
save_path.parent.mkdir(parents=True, exist_ok=True)

best_val_loss = float("inf")
patience = 5
wait = 0

print(f"Training LSTM on: {DEVICE}")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for X, y in train_loader:
        X, y = X.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            val_loss += criterion(model(X), y).item()

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)

    scheduler.step(val_loss)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train: {train_loss:.4f} | Val: {val_loss:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        wait = 0
        torch.save(model.state_dict(), save_path)
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping.")
            break

# Load best checkpoint
model.load_state_dict(
    torch.load(save_path, map_location=DEVICE, weights_only=True)
)
model.eval()

# Tune classification threshold using validation data only
val_probs, val_actual = [], []

with torch.no_grad():
    for X, y in val_loader:
        outputs = model(X.to(DEVICE))
        probs = torch.softmax(outputs, dim=1)[:, 1]

        val_probs.extend(probs.cpu().numpy())
        val_actual.extend(y.numpy())

best_threshold, best_f1 = 0.5, -1

for threshold in np.arange(0.30, 0.71, 0.02):
    predictions = (np.array(val_probs) >= threshold).astype(int)
    score = f1_score(val_actual, predictions, zero_division=0)

    if score > best_f1:
        best_f1 = score
        best_threshold = threshold

print(f"Best validation threshold: {best_threshold:.2f}")
print(f"Validation PLANET F1: {best_f1:.4f}")

# Final evaluation on untouched test data
actual, predicted = [], []

with torch.no_grad():
    for X, y in test_loader:
        outputs = model(X.to(DEVICE))
        probs = torch.softmax(outputs, dim=1)[:, 1]

        predicted.extend(
            (probs.cpu().numpy() >= best_threshold).astype(int)
        )
        actual.extend(y.numpy())

print("\nLSTM NON-ZERO CURVE TEST RESULTS")

print(classification_report(
    actual,
    predicted,
    labels=[0, 1],
    target_names=["FALSE_POSITIVE", "PLANET"],
    zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(actual, predicted))

print(f"\nSaved LSTM model: {save_path}")
