"""
AI Mock Interview Module
---------------------------
Uses an LLM (via LangChain) to:
1. Generate role-specific interview questions
2. Analyze candidate answers - relevance, filler words, STAR method usage
"""

import re
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# NOTE: requires OPENAI_API_KEY env variable. Swap for any provider LangChain supports.
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

FILLER_WORDS = ["um", "uh", "like", "you know", "basically", "actually", "i mean"]


def generate_questions(job_role: str, num_questions: int = 5) -> list:
    """Generates interview questions tailored to a job role."""
    prompt = PromptTemplate(
        input_variables=["role", "n"],
        template=(
            "You are a technical interviewer. Generate {n} interview questions "
            "for a candidate applying for the role of '{role}'. "
            "Mix technical and behavioral questions. "
            "Return only a numbered list, no extra text."
        ),
    )
    chain = prompt | llm
    response = chain.invoke({"role": job_role, "n": num_questions})

    # Parse numbered list into a clean python list
    lines = response.content.strip().split("\n")
    questions = [re.sub(r"^\d+[\.\)]\s*", "", line).strip() for line in lines if line.strip()]
    return questions


def count_filler_words(answer_text: str) -> dict:
    text_lower = answer_text.lower()
    counts = {word: text_lower.count(word) for word in FILLER_WORDS if word in text_lower}
    return counts


def check_star_method(answer_text: str) -> dict:
    """
    Heuristic check for STAR method (Situation, Task, Action, Result)
    by looking for related keywords/phrases in the answer.
    """
    indicators = {
        "Situation": ["situation", "context", "when i was", "at my previous", "faced"],
        "Task": ["task", "goal", "needed to", "responsible for", "objective"],
        "Action": ["i did", "i implemented", "i decided", "action", "approach", "i built"],
        "Result": ["result", "outcome", "achieved", "led to", "improved", "impact"],
    }
    text_lower = answer_text.lower()
    found = {}
    for component, keywords in indicators.items():
        found[component] = any(kw in text_lower for kw in keywords)
    return found


def evaluate_answer(question: str, answer_text: str) -> dict:
    """
    Full evaluation: uses LLM for relevance/content feedback,
    plus rule-based checks for filler words and STAR structure.
    """
    prompt = PromptTemplate(
        input_variables=["question", "answer"],
        template=(
            "Question: {question}\n"
            "Candidate's Answer: {answer}\n\n"
            "As an interview coach, give a short (3-4 sentence) constructive "
            "feedback on this answer - assess relevance, clarity, and depth. "
            "Then give a score out of 10."
        ),
    )
    chain = prompt | llm
    llm_feedback = chain.invoke({"question": question, "answer": answer_text}).content

    return {
        "llm_feedback": llm_feedback,
        "filler_words": count_filler_words(answer_text),
        "star_method_coverage": check_star_method(answer_text),
        "word_count": len(answer_text.split()),
    }


if __name__ == "__main__":
    qs = generate_questions("Machine Learning Engineer", num_questions=3)
    print(qs)

    sample_answer = "Um, so basically I was working on a project where the task was to improve model accuracy. I implemented feature engineering and the result was a 15% improvement."
    feedback = evaluate_answer(qs[0], sample_answer)
    print(feedback)
