
from fastapi.testclient import TestClient

from backend.main import app as backend_app
from services.ml_service.main import app as ml_app


backend_client = TestClient(backend_app)
ml_client = TestClient(ml_app)


def test_backend_home():
    response = backend_client.get("/")
    assert response.status_code == 200
    assert response.json()["project"] == "New Worlds"


def test_backend_health():
    response = backend_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_ml_service_health():
    response = ml_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_ml_planet_input():
    response = ml_client.post(
        "/planet/input-check",
        json={
            "planet_radius": 1.2,
            "planet_mass": 1.5,
            "equilibrium_temperature": 280,
            "orbital_period": 365,
            "insolation_flux": 1.0
        }
    )

    assert response.status_code == 200
    assert response.json()["model_status"] == "Not trained yet"
