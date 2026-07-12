"""
Explainable ATS Scorer
------------------------
Unlike black-box ATS tools, this module returns WHY a resume scored
a certain way - matched skills, missing skills, experience gap etc.
"""

from typing import Dict, List


def calculate_ats_score(
    candidate_skills: List[str],
    required_skills: List[str],
    candidate_experience: float,
    required_experience: float,
    candidate_education: List[str],
    required_education: List[str] = None,
) -> Dict:
    """
    Returns a score (0-100) plus a full explanation breakdown.
    Weighting: Skills 60%, Experience 30%, Education 10%
    """
    required_education = required_education or []

    # ---- Skills match (60%) ----
    candidate_skills_lower = set(s.lower() for s in candidate_skills)
    required_skills_lower = set(s.lower() for s in required_skills)

    matched_skills = candidate_skills_lower & required_skills_lower
    missing_skills = required_skills_lower - candidate_skills_lower

    skills_score = (len(matched_skills) / len(required_skills_lower) * 60) \
        if required_skills_lower else 60

    # ---- Experience match (30%) ----
    if required_experience > 0:
        exp_ratio = min(candidate_experience / required_experience, 1.0)
    else:
        exp_ratio = 1.0
    experience_score = exp_ratio * 30

    # ---- Education match (10%) ----
    if required_education:
        edu_match = any(e.lower() in [x.lower() for x in candidate_education] for e in required_education)
        education_score = 10 if edu_match else 0
    else:
        education_score = 10  # no specific requirement = full marks

    total_score = round(skills_score + experience_score + education_score, 1)

    explanation = {
        "total_score": total_score,
        "breakdown": {
            "skills_score": round(skills_score, 1),
            "experience_score": round(experience_score, 1),
            "education_score": round(education_score, 1),
        },
        "matched_skills": sorted(matched_skills),
        "missing_skills": sorted(missing_skills),
        "experience_gap": max(0, round(required_experience - candidate_experience, 1)),
        "verdict": get_verdict(total_score),
        "reasoning": generate_reasoning(
            total_score, matched_skills, missing_skills,
            candidate_experience, required_experience
        ),
    }
    return explanation


def get_verdict(score: float) -> str:
    if score >= 80:
        return "Strong Match"
    elif score >= 60:
        return "Moderate Match"
    elif score >= 40:
        return "Weak Match"
    else:
        return "Poor Match"


def generate_reasoning(score, matched, missing, cand_exp, req_exp) -> str:
    """Human-readable explanation - this is the 'explainability' USP."""
    reasoning = f"This resume scored {score}%. "

    if matched:
        reasoning += f"Strong points: matched {len(matched)} required skills " \
                      f"({', '.join(list(matched)[:5])}). "

    if missing:
        reasoning += f"Gaps: missing {len(missing)} required skills " \
                      f"({', '.join(list(missing)[:5])}). "

    if cand_exp < req_exp:
        reasoning += f"Experience gap: candidate has {cand_exp} years, " \
                      f"role requires {req_exp} years."
    else:
        reasoning += "Experience requirement is met."

    return reasoning


if __name__ == "__main__":
    result = calculate_ats_score(
        candidate_skills=["python", "sql", "pandas", "fastapi"],
        required_skills=["python", "sql", "aws", "docker"],
        candidate_experience=2.5,
        required_experience=3,
        candidate_education=["b.tech"],
    )
    print(result)
