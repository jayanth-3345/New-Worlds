from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from predict import predict_transit, predict_transit_lstm

app = FastAPI(title="New Worlds - Deep Learning Service", version="2.0.0")

class TransitRequest(BaseModel):
    flux: List[float] = Field(..., min_length=201, max_length=201)

@app.get("/")
def home():
    return {"service": "New Worlds DL Service", "models": ["TransitCNN", "TransitLSTM"]}

@app.get("/health")
def health():
    return {"status": "healthy", "cnn_loaded": True, "lstm_loaded": True}

@app.post("/transit/predict")
def predict_cnn(request: TransitRequest):
    try:
        return predict_transit(request.flux)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

@app.post("/transit/predict/lstm")
def predict_lstm(request: TransitRequest):
    try:
        return predict_transit_lstm(request.flux)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
