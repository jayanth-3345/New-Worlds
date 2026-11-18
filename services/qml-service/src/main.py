
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from optimizer import classical_scheduler, quantum_scheduler, vqe_scheduler

app = FastAPI(
    title="New Worlds - QML Observation Scheduler",
    description="Classical and QAOA-based telescope observation scheduling.",
    version="1.0.0",
)


class Target(BaseModel):
    name: str
    duration: float = Field(gt=0)
    priority: float = Field(ge=0)


class ScheduleRequest(BaseModel):
    targets: list[Target] = Field(min_length=1, max_length=6)
    available_time: float = Field(ge=0)
    steps: int = Field(default=100, ge=1, le=500)


@app.get("/")
def home():
    return {
        "service": "New Worlds QML Scheduler",
        "status": "running",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}



@app.post("/schedule/compare")
def compare_schedules(request: ScheduleRequest):
    targets = [
        target.model_dump()
        for target in request.targets
    ]

    try:
        classical = classical_scheduler(
            targets, request.available_time
        )

        quantum = quantum_scheduler(
            targets, request.available_time, request.steps
        )

        vqe = vqe_scheduler(
            targets, request.available_time, request.steps
        )

        return {
            "classical_result": classical,
            "quantum_result": quantum,
            "vqe_result": vqe,
            "comparison": {
                "qaoa_matches_classical": (
                    classical["total_priority"]
                    == quantum["total_priority"]
                ),
                "vqe_matches_classical": (
                    classical["total_priority"]
                    == vqe["total_priority"]
                ),
                "qaoa_method": "PennyLane QAOA",
                "vqe_method": "PennyLane VQE",
                "quantum_steps": request.steps,
                "quantum_advantage_claimed": False,
            },
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )
