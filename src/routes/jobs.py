from fastapi import APIRouter, HTTPException

from database import get_job_db

router = APIRouter()


@router.get(
    "/jobs",
    summary="List recent pull jobs",
    description="Returns the 50 most recent pull jobs (any status).",
    tags=["jobs"],
)
async def list_jobs():
    jobs = get_job_db().list_jobs(limit=50)
    return {"jobs": [{
        "id": j["id"], "model": j["model"], "status": j["status"],
        "created_at": j.get("created_at"), "started_at": j.get("started_at"),
        "finished_at": j.get("finished_at"),
    } for j in jobs]}


@router.get(
    "/jobs/{job_id}",
    summary="Get job status",
    description=(
        "Returns full job state including `progress`, `downloaded_bytes`, "
        "`total_bytes`, `error`, and `traceback`."
    ),
    tags=["jobs"],
    responses={
        200: {"description": "Job found."},
        404: {"description": "No job with that id."},
    },
)
async def get_job(job_id: str):
    job = get_job_db().get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job
