# 🎯 CareerLens AI — Explainable AI Recruitment Platform

An end-to-end AI recruitment system that goes beyond simple keyword-matching
ATS tools. CareerLens explains **why** a candidate scored the way they did,
matches resumes to jobs using semantic (RAG) search, runs AI-powered mock
interviews with structured feedback, and flags potential bias in candidate
rankings.

## Why this project is different
Most "AI resume checker" projects do keyword matching and stop there.
CareerLens adds:
- **Explainability** — every ATS score comes with a reasoning breakdown, not just a number
- **Semantic matching (RAG)** — matches meaning, not just exact keywords, using FAISS + sentence embeddings
- **Responsible AI angle** — a bias/fairness checker on candidate rankings
- **Structured interview feedback** — checks for STAR method usage and filler words, not just "good answer / bad answer"

## Architecture
```
Streamlit Frontend  →  FastAPI Backend  →  SQLite/PostgreSQL
                              │
        ┌─────────────┬──────┴──────┬─────────────┐
        ▼             ▼             ▼             ▼
  Resume Parser   ATS Scorer   RAG Job Matcher  Bias Checker
   (spaCy NLP)   (explainable)  (FAISS + LLM)   (fairness)
```

## Tech Stack
Python · FastAPI · Streamlit · spaCy · Scikit-learn · LangChain ·
FAISS · Sentence-Transformers · SQLAlchemy · Docker

## Project Structure
```
careerlens-ai/
├── app/
│   ├── main.py                  # FastAPI entrypoint
│   ├── modules/
│   │   ├── resume_parser.py     # PDF/DOCX parsing + NER extraction
│   │   ├── ats_scorer.py        # Explainable scoring logic
│   │   ├── job_matcher.py       # RAG / FAISS semantic matching
│   │   ├── mock_interview.py    # LLM-based Q&A + feedback
│   │   └── bias_checker.py      # Fairness auditing on rankings
│   └── database/
│       └── models.py            # SQLAlchemy models
├── data/
│   └── sample_jobs.csv          # Sample job postings for FAISS index
├── streamlit_app.py             # Frontend UI
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Setup — Local Development

1. **Clone & install dependencies**
```bash
git clone <your-repo-url>
cd careerlens-ai
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

2. **Set your LLM API key** (needed for mock_interview.py)
```bash
export OPENAI_API_KEY="your-key-here"   # Windows: set OPENAI_API_KEY=your-key-here
```

3. **Run the backend**
```bash
uvicorn app.main:app --reload
```
Visit `http://localhost:8000/docs` for interactive API docs (Swagger UI).

4. **Run the frontend** (in a new terminal)
```bash
streamlit run streamlit_app.py
```
Visit `http://localhost:8501`

## Setup — Docker

```bash
docker-compose up --build
```
- Backend: `http://localhost:8000`
- Frontend: `http://localhost:8501`

## Testing Individual Modules
Every module has a `if __name__ == "__main__":` block so you can test it
standalone before wiring it into the API:
```bash
python app/modules/ats_scorer.py
python app/modules/job_matcher.py
python app/modules/bias_checker.py
```

## Resume Bullet Point (for your CV)
> Built CareerLens AI, an explainable AI recruitment platform using RAG
> (LangChain + FAISS + Sentence-Transformers) for semantic resume-job
> matching, an explainable ATS scoring engine, AI-driven mock interviews
> with NLP-based feedback, and a fairness/bias-checking module for
> candidate rankings. Backend built with FastAPI + SQLAlchemy, frontend
> with Streamlit, containerized with Docker.

## Roadmap / Future Improvements
- Replace name-based bias heuristic with proper demographic-blind auditing
- Add authentication (recruiter vs candidate roles)
- Fine-tune embeddings on domain-specific resume/JD pairs
- Add PDF report export for ATS results
- Deploy on AWS/Render with PostgreSQL

## Disclaimer
The bias-checking module is a **demo-level heuristic** built to showcase
responsible-AI awareness. It is not a statistically validated fairness
audit and should not be used for real hiring decisions without proper
review.
