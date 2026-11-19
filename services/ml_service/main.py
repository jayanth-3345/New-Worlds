from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import skops.io as sio

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from xgboost import XGBClassifier

app = FastAPI(
    title="New Worlds - ML Service",
    description="Random Forest and XGBoost Exoplanet Classification",
    version="3.0.0"
)

ROOT = Path(__file__).resolve().parents[2]

RF_PATH = ROOT / "models/model.skops"
XGB_PATH = ROOT / "models/xgboost_model.ubj"
IMPUTER_PATH = ROOT / "models/xgboost_imputer.joblib"

FEATURES = [
    "pl_rade", "pl_eqt", "pl_insol",
    "st_teff", "st_mass", "st_rad"
]

rf_model = None
xgb_model = None
xgb_imputer = None


class PlanetData(BaseModel):
    planet_radius: float = Field(gt=0)
    equilibrium_temperature: float = Field(gt=0)
    insolation_flux: float = Field(gt=0)
    star_temperature: float = Field(gt=0)
    star_mass: float = Field(gt=0)
    star_radius: float = Field(gt=0)


@app.on_event("startup")
def load_models():
    global rf_model, xgb_model, xgb_imputer

    rf_model = sio.load(
        RF_PATH,
        trusted=["numpy.dtype", "sklearn.tree._tree.Tree"]
    )

    xgb_model = XGBClassifier()
    xgb_model.load_model(str(XGB_PATH))

    xgb_imputer = joblib.load(IMPUTER_PATH)


def get_features(planet):
    return pd.DataFrame([{
        "pl_rade": planet.planet_radius,
        "pl_eqt": planet.equilibrium_temperature,
        "pl_insol": planet.insolation_flux,
        "st_teff": planet.star_temperature,
        "st_mass": planet.star_mass,
        "st_rad": planet.star_radius
    }], columns=FEATURES)


def result(planet, prediction, model_name):
    prediction = int(prediction)
    return {
        "planet_data": planet.model_dump(),
        "prediction": prediction,
        "classification": (
            "Potentially Habitable" if prediction == 1
            else "Not Classified as Potentially Habitable"
        ),
        "model": model_name,
        "disclaimer": (
            "Prediction reflects catalog-derived training labels, "
            "not confirmation of life or actual habitability."
        )
    }


@app.get("/")
def home():
    return {
        "service": "ML Service",
        "models": ["Random Forest", "XGBoost"]
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "random_forest_loaded": rf_model is not None,
        "xgboost_loaded": xgb_model is not None
    }


@app.post("/planet/predict")
def predict_random_forest(planet: PlanetData):
    if rf_model is None:
        raise HTTPException(status_code=503, detail="Random Forest not loaded")

    prediction = rf_model.predict(get_features(planet))[0]
    return result(planet, prediction, "Random Forest")


@app.post("/planet/predict/xgboost")
def predict_xgboost(planet: PlanetData):
    if xgb_model is None or xgb_imputer is None:
        raise HTTPException(status_code=503, detail="XGBoost not loaded")

    features = xgb_imputer.transform(get_features(planet))
    prediction = xgb_model.predict(features)[0]
    return result(planet, prediction, "XGBoost")
