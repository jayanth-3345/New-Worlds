from pathlib import Path
import numpy as np
import torch

from model import TransitCNN

BASE = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE / "models" / "transit_cnn.pth"

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

THRESHOLD = 0.56

model = TransitCNN(num_classes=2).to(DEVICE)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
)

model.eval()


def predict_transit(flux):
    flux = np.asarray(flux, dtype=np.float32)

    if flux.shape != (201,):
        raise ValueError("Exactly 201 flux values are required.")

    if not np.isfinite(flux).all():
        raise ValueError("Flux values must be finite.")

    if np.all(flux == 0):
        return {
            "prediction": 0,
            "classification": "NO_SIGNAL",
            "confidence": 1.0
        }

    median = np.median(flux)

    if median <= 0:
        raise ValueError("Flux median must be positive.")

    normalized = (flux / median - 1.0) * 1000.0

    tensor = torch.tensor(
        normalized, dtype=torch.float32
    ).reshape(1, 1, 201).to(DEVICE)

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]

    planet_probability = probabilities[1].item()

    if planet_probability >= THRESHOLD:
        classification = "PLANET"
        confidence = planet_probability
        prediction = 2
    else:
        classification = "FALSE_POSITIVE"
        confidence = 1 - planet_probability
        prediction = 1

    return {
        "prediction": prediction,
        "classification": classification,
        "confidence": round(confidence, 4)
    }