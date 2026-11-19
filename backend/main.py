import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx

app = FastAPI(
    title="New Worlds API",
    description="Exoplanet Analysis and Observation Scheduling System",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://127.0.0.1:8002/planet/predict")
DL_SERVICE_URL = os.getenv("DL_SERVICE_URL", "http://127.0.0.1:8003/transit/predict")
QML_SERVICE_URL = os.getenv("QML_SERVICE_URL", "http://127.0.0.1:8004/schedule/compare")
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://127.0.0.1:8005/schedule")


class PlanetData(BaseModel):
    planet_radius: float = Field(gt=0)
    equilibrium_temperature: float = Field(gt=0)
    insolation_flux: float = Field(gt=0)
    star_temperature: float = Field(gt=0)
    star_mass: float = Field(gt=0)
    star_radius: float = Field(gt=0)

class TransitData(BaseModel):
    flux: list[float] = Field(min_length=201, max_length=201)

class ObservationTarget(BaseModel):
    name: str
    duration: float = Field(gt=0)
    priority: float = Field(ge=0)


class ScheduleData(BaseModel):
    targets: list[ObservationTarget] = Field(min_length=1, max_length=6)
    available_time: float = Field(ge=0)
    steps: int = Field(default=100, ge=1, le=500)


@app.get("/")
def home():
    return {
        "project": "New Worlds",
        "message": "Backend is running successfully!"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "project": "New Worlds"
    }


@app.post("/planet/predict")
async def predict_planet(planet: PlanetData):
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                ML_SERVICE_URL,
                json=planet.model_dump()
            )

        response.raise_for_status()
        return response.json()

    except httpx.HTTPError as error:
        raise HTTPException(
            status_code=502,
            detail=f"ML service error: {str(error)}"
        )
@app.post("/transit/predict")
async def predict_transit(data: TransitData):
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                DL_SERVICE_URL,
                json=data.model_dump()
            )

        response.raise_for_status()
        return response.json()

    except httpx.HTTPError as error:
        raise HTTPException(
            status_code=502,
            detail=f"DL service error: {str(error)}"
        )
@app.post("/schedule/compare")
async def compare_schedules(data: ScheduleData):
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                QML_SERVICE_URL,
                json=data.model_dump()
            )

        response.raise_for_status()
        return response.json()

    except httpx.HTTPError as error:
        raise HTTPException(
            status_code=502,
            detail=f"QML service error: {str(error)}"
        )
@app.post("/llm/schedule")
async def llm_schedule(request: ScheduleData):
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post(
                LLM_SERVICE_URL,
                json=request.model_dump()
            )
            response.raise_for_status()
            return response.json()

        except httpx.HTTPError as error:
            raise HTTPException(
                status_code=502,
                detail=f"LLM service error: {str(error)}"
            )
@app.post("/planet/predict/xgboost")
async def predict_planet_xgboost(planet: PlanetData):
    try:
        xgboost_url = ML_SERVICE_URL.replace(
            "/planet/predict", "/planet/predict/xgboost"
        )
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                xgboost_url,
                json=planet.model_dump()
            )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as error:
        raise HTTPException(
            status_code=502,
            detail=f"XGBoost service error: {str(error)}"
        )

@app.post("/transit/predict/lstm")
async def predict_transit_lstm(data: TransitData):
    try:
        lstm_url = DL_SERVICE_URL.replace(
            "/transit/predict", "/transit/predict/lstm"
        )
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                lstm_url,
                json=data.model_dump()
            )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as error:
        raise HTTPException(
            status_code=502,
            detail=f"LSTM service error: {str(error)}"
        )
