"""
Bias & Fairness Checker
--------------------------
Analyzes candidate rankings to flag potential bias patterns -
e.g., score disparities correlated with gender-coded names,
name origin, age indicators (graduation year), etc.

NOTE: This is a heuristic/demo-level fairness checker meant to
showcase "responsible AI" awareness. Production fairness auditing
requires proper demographic data collection with consent and
statistical significance testing (e.g., disparate impact ratio).
"""

import pandas as pd
from typing import List, Dict

# Simple demo name lists - in a real system, avoid inferring gender/ethnicity
# from names directly; this is for illustrative bias-awareness purposes only.
COMMONLY_MALE_NAMES = {"james", "john", "robert", "michael", "david", "arjun", "rahul", "aman"}
COMMONLY_FEMALE_NAMES = {"mary", "jennifer", "linda", "priya", "anjali", "sneha", "divya"}


def infer_group(name: str) -> str:
    first_name = name.lower().split()[0] if name else ""
    if first_name in COMMONLY_MALE_NAMES:
        return "Group A"
    elif first_name in COMMONLY_FEMALE_NAMES:
        return "Group B"
    return "Unknown"


def check_ranking_bias(candidates: List[Dict]) -> Dict:
    """
    candidates: list of dicts like [{"name": "...", "ats_score": 85}, ...]
    Returns average score per inferred group + a fairness flag.
    """
    df = pd.DataFrame(candidates)
    df["group"] = df["name"].apply(infer_group)

    group_stats = df.groupby("group")["ats_score"].agg(["mean", "count"]).to_dict("index")

    # Disparate impact style check: flag if group averages differ by > 15 points
    means = [v["mean"] for v in group_stats.values() if v["count"] > 0]
    bias_flag = False
    if len(means) >= 2:
        bias_flag = (max(means) - min(means)) > 15

    return {
        "group_statistics": group_stats,
        "potential_bias_detected": bias_flag,
        "note": (
            "Significant score disparity detected between groups - recommend "
            "manual review of scoring criteria."
            if bias_flag else
            "No significant disparity detected in this sample."
        ),
        "disclaimer": (
            "This is a heuristic demo check based on name patterns, not a "
            "statistically rigorous fairness audit. Real-world bias auditing "
            "requires proper demographic data and significance testing."
        ),
    }


if __name__ == "__main__":
    sample_candidates = [
        {"name": "James Smith", "ats_score": 88},
        {"name": "Mary Johnson", "ats_score": 65},
        {"name": "Rahul Kumar", "ats_score": 90},
        {"name": "Priya Sharma", "ats_score": 68},
    ]
    result = check_ranking_bias(sample_candidates)
    print(result)
