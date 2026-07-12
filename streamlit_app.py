"""
CareerLens AI - Streamlit Frontend
--------------------------------------
A simple UI on top of the FastAPI backend. Run with:
    streamlit run streamlit_app.py

Make sure the FastAPI backend is running first:
    uvicorn app.main:app --reload
"""

import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="CareerLens AI", page_icon="🎯", layout="wide")
st.title("🎯 CareerLens AI - Explainable Recruitment Assistant")

tab1, tab2, tab3, tab4 = st.tabs(
    ["📄 Upload Resume", "✅ ATS Score", "🔍 Job Matching", "⚖️ Bias Check"]
)

# ---------------- TAB 1: Upload Resume ----------------
with tab1:
    st.header("Upload Resume")
    uploaded_file = st.file_uploader("Choose a resume (PDF/DOCX)", type=["pdf", "docx"])

    if uploaded_file and st.button("Parse Resume"):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
        response = requests.post(f"{API_URL}/upload-resume/", files=files)

        if response.status_code == 200:
            data = response.json()
            st.session_state["candidate_id"] = data["candidate_id"]
            st.success(f"Parsed! Candidate ID: {data['candidate_id']}")
            st.json(data["parsed_data"])
        else:
            st.error("Failed to parse resume.")

# ---------------- TAB 2: ATS Score ----------------
with tab2:
    st.header("Explainable ATS Score")
    candidate_id = st.number_input("Candidate ID", min_value=1, step=1,
                                    value=st.session_state.get("candidate_id", 1))
    required_skills = st.text_input("Required Skills (comma-separated)", "python,sql,docker")
    required_experience = st.number_input("Required Experience (years)", min_value=0.0, value=2.0)

    if st.button("Calculate ATS Score"):
        params = {"required_skills": required_skills, "required_experience": required_experience}
        response = requests.post(f"{API_URL}/ats-score/{candidate_id}", params=params)

        if response.status_code == 200:
            result = response.json()
            st.metric("ATS Score", f"{result['total_score']}%", result["verdict"])
            st.write("**Reasoning:**", result["reasoning"])
            col1, col2 = st.columns(2)
            with col1:
                st.write("✅ Matched Skills:", result["matched_skills"])
            with col2:
                st.write("❌ Missing Skills:", result["missing_skills"])
        else:
            st.error("Could not calculate score. Check candidate ID.")

# ---------------- TAB 3: Job Matching (RAG) ----------------
with tab3:
    st.header("RAG-based Job Matching")
    candidate_id_jm = st.number_input("Candidate ID for matching", min_value=1, step=1, key="jm_id")

    if st.button("Find Matching Jobs"):
        response = requests.get(f"{API_URL}/match-jobs/{candidate_id_jm}", params={"top_k": 5})
        if response.status_code == 200:
            matches = response.json()["matches"]
            for m in matches:
                st.subheader(f"{m['job_title']} @ {m['company']} — {m['match_score']}% match")
                st.write(m["description"])
                st.caption(f"Required: {m['required_skills']}")
                st.divider()
        else:
            st.error("Could not fetch matches.")

# ---------------- TAB 4: Bias Check ----------------
with tab4:
    st.header("Fairness / Bias Check Across Candidates")
    if st.button("Run Bias Check"):
        response = requests.get(f"{API_URL}/check-bias/")
        if response.status_code == 200:
            result = response.json()
            st.json(result)
        else:
            st.error("Bias check failed.")
