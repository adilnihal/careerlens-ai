"""
Resume Parser Module
---------------------
Extracts structured data (skills, experience, education, contact info)
from resume files (PDF/DOCX) using spaCy NLP + regex patterns.
"""

import re
import spacy
from pypdf import PdfReader
from docx import Document
from typing import Dict, List

# Load spaCy model (run: python -m spacy download en_core_web_sm)
nlp = spacy.load("en_core_web_sm")

# A reference skill set - expand this list based on your target domain
SKILL_DATABASE = [
    "python", "java", "sql", "javascript", "react", "node.js", "fastapi",
    "django", "flask", "pandas", "numpy", "scikit-learn", "tensorflow",
    "pytorch", "nlp", "langchain", "rag", "faiss", "docker", "kubernetes",
    "git", "aws", "azure", "gcp", "streamlit", "power bi", "tableau",
    "excel", "machine learning", "deep learning", "data analysis",
    "rest api", "mongodb", "postgresql", "mysql", "linux", "html", "css"
]


def extract_text_from_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def extract_text_from_docx(file_path: str) -> str:
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])


def extract_text(file_path: str) -> str:
    if file_path.lower().endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.lower().endswith(".docx"):
        return extract_text_from_docx(file_path)
    else:
        raise ValueError("Unsupported file format. Use PDF or DOCX.")


def extract_email(text: str) -> str:
    match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    return match.group(0) if match else None


def extract_phone(text: str) -> str:
    match = re.search(r"(\+?\d{1,3}[-.\s]?)?\(?\d{3,5}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}", text)
    return match.group(0) if match else None


def extract_skills(text: str) -> List[str]:
    text_lower = text.lower()
    found_skills = [skill for skill in SKILL_DATABASE if skill in text_lower]
    return sorted(set(found_skills))


def extract_experience_years(text: str) -> float:
    """Rough heuristic: finds patterns like '3 years', '5+ years experience'."""
    matches = re.findall(r"(\d+\.?\d*)\+?\s*(?:years|yrs)", text.lower())
    if matches:
        return max(float(m) for m in matches)
    return 0.0


def extract_education(text: str) -> List[str]:
    edu_keywords = ["b.tech", "btech", "bachelor", "m.tech", "mtech", "master",
                     "b.sc", "bsc", "m.sc", "msc", "mba", "phd", "diploma"]
    text_lower = text.lower()
    found = [kw for kw in edu_keywords if kw in text_lower]
    return sorted(set(found))



# Common non-name words that spaCy sometimes misclassifies as PERSON
# (states, cities, generic terms that appear near the top of resumes)
NAME_BLOCKLIST = {
    "kerala", "india", "resume", "curriculum vitae", "cv", "karnataka",
    "tamil nadu", "maharashtra", "delhi", "mumbai", "bangalore", "chennai",
    "hyderabad", "kochi", "kozhikode", "thiruvananthapuram",
}

# If any of these words appear in a candidate entity, it's very likely a
# job title / skill heading, not a person's name (e.g. "Generative AI",
# "Machine Learning Engineer")
JOB_TITLE_KEYWORDS = {
    "ai", "ml", "engineer", "developer", "generative", "data", "science",
    "scientist", "analyst", "intern", "manager", "consultant", "architect",
    "specialist", "software", "backend", "frontend", "full stack",
    "learning", "artificial", "intelligence", "objective", "summary",
}


def _looks_like_job_title(candidate: str) -> bool:
    words = set(candidate.lower().replace(".", "").split())
    return bool(words & JOB_TITLE_KEYWORDS)


def _first_line_name_guess(text: str) -> str:
    """
    Fallback heuristic: many resumes have the candidate's name as the very
    first non-empty line, in Title Case, with no digits/emails/keywords.
    """
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        if "@" in line or any(ch.isdigit() for ch in line):
            continue
        if _looks_like_job_title(line):
            continue
        words = line.split()
        # Require at least 2 words - a single word is too risky (could be
        # a truncated word from column-wrapped PDF text, e.g. "Proficien")
        if 2 <= len(words) <= 4 and all(w.replace(".", "").isalpha() for w in words):
            return line
        break  # only check the very first non-empty line
    return "Unknown"


def _filename_name_guess(filename: str) -> str:
    """
    Fallback heuristic: many people name their resume file after
    themselves, e.g. 'Adil_Nihal_AI_ML_Engineer.pdf' -> 'Adil Nihal'.
    Strips extension, splits on _/-, drops job-title keywords and
    all-caps abbreviations (AI, ML, CV etc.), keeps the leading name parts.
    """
    import os
    stem = os.path.splitext(os.path.basename(filename))[0]
    stem = re.sub(r"\(\d+\)$", "", stem).strip()  # drop trailing "(1)" etc.
    parts = re.split(r"[_\-\s]+", stem)

    name_parts = []
    for part in parts:
        if not part.isalpha():
            continue
        if part.lower() in JOB_TITLE_KEYWORDS or part.lower() in NAME_BLOCKLIST:
            break  # stop at the first job-title-looking word
        if part.isupper() and len(part) <= 3:
            break  # things like "AI", "ML", "CV"
        name_parts.append(part.capitalize())
        if len(name_parts) == 2:  # first + last name is usually enough
            break

    return " ".join(name_parts) if name_parts else "Unknown"


def extract_name(text: str, filename: str = None) -> str:
    """
    Uses spaCy NER to guess the candidate's name. Picks the first PERSON
    entity that isn't in the blocklist, doesn't look like a job title,
    and looks like a plausible name (2+ words, mostly alphabetic).
    Falls back to: (1) filename heuristic, (2) first-line heuristic.
    PDF text extraction order can get jumbled - and words can get cut
    off mid-word - for resumes with multi-column or sidebar designs,
    so the filename is tried before the riskier first-line guess.
    """
    doc = nlp(text[:500])  # name usually appears near the top
    for ent in doc.ents:
        if ent.label_ != "PERSON":
            continue
        candidate = ent.text.strip()
        if candidate.lower() in NAME_BLOCKLIST:
            continue
        if _looks_like_job_title(candidate):
            continue
        if len(candidate) > 40 or len(candidate.split()) > 4:
            continue  # too long to be a real name
        if len(candidate.split()) < 2:
            continue  # single-word "names" are too risky (often truncated
                       # words from column-wrapped PDFs, misclassified by NER)
        if not all(word.replace(".", "").isalpha() for word in candidate.split()):
            continue  # contains numbers/symbols, unlikely to be a name
        return candidate

    if filename:
        filename_guess = _filename_name_guess(filename)
        if filename_guess != "Unknown":
            return filename_guess

    return _first_line_name_guess(text)


def parse_resume(file_path: str) -> Dict:
    """Main function: returns a structured dictionary of parsed resume data."""
    text = extract_text(file_path)

    parsed_data = {
        "name": extract_name(text, filename=file_path),
        "email": extract_email(text),
        "phone": extract_phone(text),
        "skills": extract_skills(text),
        "experience_years": extract_experience_years(text),
        "education": extract_education(text),
        "raw_text": text,
    }
    return parsed_data


if __name__ == "__main__":
    # Quick test
    sample = parse_resume("sample_resume.pdf")
    print(sample)
