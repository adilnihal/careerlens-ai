"""
RAG-based Job Matching Engine
-------------------------------
Uses sentence embeddings + FAISS to do SEMANTIC matching between
resumes and job descriptions (not just keyword matching).

Example: resume says "data wrangling", JD says "data cleaning"
-> keyword match fails, but semantic match succeeds because
   the embeddings capture that these mean the same thing.
"""

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import List, Dict

# Lightweight, fast embedding model - good for semantic similarity tasks
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")


class JobMatcher:
    def __init__(self, jobs_csv_path: str = None):
        self.index = None
        self.jobs_df = None
        self.embeddings = None
        if jobs_csv_path:
            self.load_jobs(jobs_csv_path)

    def load_jobs(self, jobs_csv_path: str):
        """Loads job postings and builds the FAISS index."""
        self.jobs_df = pd.read_csv(jobs_csv_path)
        # Combine title + description + required_skills into one text blob per job
        job_texts = (
            self.jobs_df["title"] + ". " +
            self.jobs_df["description"] + ". Required skills: " +
            self.jobs_df["required_skills"]
        ).tolist()

        self.embeddings = EMBED_MODEL.encode(job_texts, show_progress_bar=False)
        self.embeddings = np.array(self.embeddings).astype("float32")

        # Normalize for cosine similarity via inner product
        faiss.normalize_L2(self.embeddings)

        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product = cosine sim (normalized)
        self.index.add(self.embeddings)

    def find_matching_jobs(self, resume_text: str, top_k: int = 5) -> List[Dict]:
        """Given resume text, returns top_k most semantically similar jobs."""
        if self.index is None:
            raise ValueError("Job index not built. Call load_jobs() first.")

        query_embedding = EMBED_MODEL.encode([resume_text]).astype("float32")
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            job = self.jobs_df.iloc[idx]
            results.append({
                "job_title": job["title"],
                "company": job.get("company", "N/A"),
                "match_score": round(float(score) * 100, 1),  # cosine similarity -> %
                "required_skills": job["required_skills"],
                "description": job["description"][:200] + "...",
            })
        return results


if __name__ == "__main__":
    matcher = JobMatcher("../../data/sample_jobs.csv")
    resume_sample = """
    Experienced Python developer with strong background in data wrangling,
    building REST APIs using FastAPI, and deploying ML models with Docker.
    """
    matches = matcher.find_matching_jobs(resume_sample, top_k=3)
    for m in matches:
        print(m)
