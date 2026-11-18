from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from predict import predict_transit

app = FastAPI(
    title="New Worlds - Deep Learning Service",
    version="1.0.0"
)


class TransitRequest(BaseModel):
    flux: List[float] = Field(
        ...,
        min_length=201,
        max_length=201,
        description="201 normalized telescope flux measurements"
    )


@app.get("/")
def home():
    return {
        "service": "New Worlds DL Service",
        "model": "TransitCNN V1",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "TransitCNN V1"
    }


@app.post("/transit/predict")
def predict(request: TransitRequest):
    try:
        return predict_transit(request.flux)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))