
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="New Worlds API",
    description="Exoplanet Analysis and Observation Scheduling System",
    version="1.0.0"
)

# Allow the React frontend to communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
