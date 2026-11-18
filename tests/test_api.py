
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
    with TestClient(ml_app) as client:
        response = client.post(
            "/planet/predict",
            json={
                "planet_radius": 1.2,
                "equilibrium_temperature": 280,
                "insolation_flux": 1.0,
                "star_temperature": 5778,
                "star_mass": 1.0,
                "star_radius": 1.0
            }
        )

        assert response.status_code == 200

        data = response.json()

        assert "prediction" in data
        assert "classification" in data
        assert data["model"] == "Random Forest"
