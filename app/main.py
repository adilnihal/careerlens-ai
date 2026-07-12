"""
CareerLens AI - Main FastAPI Backend
---------------------------------------
Exposes REST endpoints that tie together all modules:
resume parsing -> ATS scoring -> job matching -> bias checking -> interview.
"""

import os
import shutil
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session

from app.modules.resume_parser import parse_resume
from app.modules.ats_scorer import calculate_ats_score
from app.modules.job_matcher import JobMatcher
from app.modules.bias_checker import check_ranking_bias
from app.database.models import Candidate, init_db, get_db

app = FastAPI(title="CareerLens AI", version="1.0")

# Initialize DB tables on startup
init_db()

# Load the job matcher once at startup (builds FAISS index)
job_matcher = JobMatcher(jobs_csv_path="data/sample_jobs.csv")

UPLOAD_DIR = "uploaded_resumes"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
def health_check():
    return {"status": "CareerLens AI backend is running"}


@app.post("/upload-resume/")
async def upload_resume(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Upload a resume -> parse it -> save candidate record -> return parsed data."""
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    parsed = parse_resume(file_path)

    candidate = Candidate(
        name=parsed["name"],
        email=parsed["email"],
        phone=parsed["phone"],
        skills=parsed["skills"],
        experience_years=parsed["experience_years"],
        education=parsed["education"],
        resume_raw_text=parsed["raw_text"],
    )
    db.add(candidate)
    db.commit()
    db.refresh(candidate)

    return {"candidate_id": candidate.id, "parsed_data": parsed}


@app.post("/ats-score/{candidate_id}")
def get_ats_score(
    candidate_id: int,
    required_skills: str,   # comma-separated, e.g. "python,sql,docker"
    required_experience: float,
    db: Session = Depends(get_db),
):
    """Calculate explainable ATS score for a candidate against a job's requirements."""
    candidate = db.query(Candidate).filter(Candidate.id == candidate_id).first()
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")

    result = calculate_ats_score(
        candidate_skills=candidate.skills,
        required_skills=required_skills.split(","),
        candidate_experience=candidate.experience_years,
        required_experience=required_experience,
        candidate_education=candidate.education,
    )

    candidate.ats_score = result["total_score"]
    db.commit()

    return result


@app.get("/match-jobs/{candidate_id}")
def match_jobs(candidate_id: int, top_k: int = 5, db: Session = Depends(get_db)):
    """RAG-based semantic job matching for a given candidate."""
    candidate = db.query(Candidate).filter(Candidate.id == candidate_id).first()
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")

    matches = job_matcher.find_matching_jobs(candidate.resume_raw_text, top_k=top_k)
    return {"candidate_id": candidate_id, "matches": matches}


@app.get("/check-bias/")
def check_bias(db: Session = Depends(get_db)):
    """Runs a fairness check across all scored candidates."""
    candidates = db.query(Candidate).filter(Candidate.ats_score.isnot(None)).all()
    candidate_list = [{"name": c.name, "ats_score": c.ats_score} for c in candidates]

    if len(candidate_list) < 2:
        return {"message": "Not enough scored candidates to run a bias check."}

    return check_ranking_bias(candidate_list)


@app.get("/candidates/")
def list_candidates(db: Session = Depends(get_db)):
    candidates = db.query(Candidate).all()
    return candidates
