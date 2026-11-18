from pathlib import Path

import pandas as pd
import skops.io as sio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="New Worlds - ML Service",
    description="Exoplanet Habitability Classification",
    version="2.0.0"
)

MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "models/model.skops"
)

model = None


class PlanetData(BaseModel):
    planet_radius: float = Field(gt=0)
    equilibrium_temperature: float = Field(gt=0)
    insolation_flux: float = Field(gt=0)
    star_temperature: float = Field(gt=0)
    star_mass: float = Field(gt=0)
    star_radius: float = Field(gt=0)


@app.on_event("startup")
def load_model():
    global model

    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found: {MODEL_PATH}")

    model = sio.load(
        MODEL_PATH,
        trusted=[
            "numpy.dtype",
            "sklearn.tree._tree.Tree"
        ]
    )


@app.get("/")
def home():
    return {
        "service": "ML Service",
        "message": "Exoplanet classification service initialized"
    }


@app.get("/health")
def health():
    return {
        "service": "ML",
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/planet/predict")
def predict_planet(planet: PlanetData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features = pd.DataFrame([{
        "pl_rade": planet.planet_radius,
        "pl_eqt": planet.equilibrium_temperature,
        "pl_insol": planet.insolation_flux,
        "st_teff": planet.star_temperature,
        "st_mass": planet.star_mass,
        "st_rad": planet.star_radius
    }])

    prediction = int(model.predict(features)[0])

    return {
        "planet_data": planet.model_dump(),
        "prediction": prediction,
        "classification": (
            "Potentially Habitable" if prediction == 1
            else "Not Classified as Potentially Habitable"
        ),
        "model": "Random Forest",
        "disclaimer": "Prediction reflects catalog-derived training labels, not confirmation of life or actual habitability."
    }
