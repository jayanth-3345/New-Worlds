from pathlib import Path
import numpy as np
import torch
from model import TransitCNN
from lstm_model import TransitLSTM

BASE = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cpu")

cnn = TransitCNN(num_classes=2).to(DEVICE)
cnn.load_state_dict(torch.load(BASE / "models/transit_cnn.pth", map_location=DEVICE, weights_only=True))
cnn.eval()

lstm = TransitLSTM(num_classes=2, hidden_size=64).to(DEVICE)
lstm.load_state_dict(torch.load(BASE / "models/transit_lstm.pth", map_location=DEVICE, weights_only=True))
lstm.eval()

THRESHOLD = 0.56

def predict_model(flux, model, name):
    flux = np.asarray(flux, dtype=np.float32)
    if flux.shape != (201,) or not np.isfinite(flux).all():
        raise ValueError("Exactly 201 finite flux values are required.")
    if np.all(flux == 0):
        return {"model": name, "prediction": 0, "classification": "NO_SIGNAL", "confidence": 1.0}

    median = np.median(flux)
    if median <= 0:
        raise ValueError("Flux median must be positive.")

    normalized = (flux / median - 1.0) * 1000.0
    tensor = torch.tensor(normalized, dtype=torch.float32).reshape(1, 1, 201).to(DEVICE)

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]

    probability = probabilities[1].item()
    prediction = 2 if probability >= THRESHOLD else 1

    return {
        "model": name,
        "prediction": prediction,
        "classification": "PLANET" if prediction == 2 else "FALSE_POSITIVE",
        "confidence": round(max(probability, 1 - probability), 4)
    }

def predict_transit(flux):
    return predict_model(flux, cnn, "TransitCNN")

def predict_transit_lstm(flux):
    return predict_model(flux, lstm, "TransitLSTM")
