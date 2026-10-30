
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="New Worlds - ML Service",
    description="Exoplanet Habitability Classification",
    version="1.0.0"
)


class PlanetData(BaseModel):
    planet_radius: float = Field(gt=0)
    planet_mass: float = Field(gt=0)
    equilibrium_temperature: float = Field(gt=0)
    orbital_period: float = Field(gt=0)
    insolation_flux: float = Field(gt=0)


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
        "status": "healthy"
    }


@app.post("/planet/input-check")
def validate_planet(planet: PlanetData):
    return {
        "message": "Planet parameters received successfully",
        "planet_data": planet.model_dump(),
        "model_status": "Not trained yet"
    }
