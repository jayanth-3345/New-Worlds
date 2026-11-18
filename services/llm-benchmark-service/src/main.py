import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Load the root .env file
ROOT_DIR = Path(__file__).resolve().parents[3]
load_dotenv(ROOT_DIR / ".env")

from benchmark import benchmark_schedule

app = FastAPI(
    title="New Worlds - LLM Benchmark Service",
    description="Groq-powered LLM observation scheduling benchmark",
    version="1.0.0"
)


class ObservationTarget(BaseModel):
    name: str
    duration: float = Field(gt=0)
    priority: float = Field(ge=0)


class ScheduleRequest(BaseModel):
    targets: list[ObservationTarget] = Field(min_length=1, max_length=6)
    available_time: float = Field(ge=0)


@app.get("/")
def home():
    return {
        "service": "New Worlds LLM Benchmark Service",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "provider": "Groq",
        "api_key_configured": bool(os.getenv("GROQ_API_KEY"))
    }


@app.post("/schedule")
def schedule(request: ScheduleRequest):
    targets = [target.model_dump() for target in request.targets]

    try:
        result = benchmark_schedule(
            targets,
            request.available_time
        )

        # Validate the LLM's selection before accepting it.
        target_map = {target["name"]: target for target in targets}
        selected = result.get("selected_targets", [])

        if not isinstance(selected, list) or any(
            name not in target_map for name in selected
        ):
            raise ValueError("LLM returned invalid target names")

        total_duration = sum(
            target_map[name]["duration"] for name in selected
        )

        total_priority = sum(
            target_map[name]["priority"] for name in selected
        )

        if total_duration > request.available_time:
            raise ValueError("LLM exceeded available telescope time")

        return {
            **result,
            "total_duration": total_duration,
            "total_priority": total_priority,
            "available_time": request.available_time
        }

    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"LLM scheduling failed: {str(error)}"
        )
