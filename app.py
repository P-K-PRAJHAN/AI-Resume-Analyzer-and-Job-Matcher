import streamlit as st

from embedding_engine import EmbeddingEngine
from llm_feedback import LLMFeedbackGenerator
from matcher import ResumeJobMatcher
from resume_parser import clean_resume_text, extract_text_from_pdf
from skill_extractor import SkillExtractor
from vector_store import ResumeVectorStore


if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False
if "ranked_results" not in st.session_state:
    st.session_state.ranked_results = []
if "job_skills" not in st.session_state:
    st.session_state.job_skills = []
if "llm_feedback" not in st.session_state:
    st.session_state.llm_feedback = ""


def run_analysis(uploaded_files, job_description: str, model_option: str):
    skill_extractor = SkillExtractor()
    matcher = ResumeJobMatcher()
    embedding_engine = EmbeddingEngine()
    vector_store = ResumeVectorStore(embedding_dim=384)
    feedback_generator = LLMFeedbackGenerator(model_name=model_option)

    cleaned_job = clean_resume_text(job_description)
    job_skills = skill_extractor.extract_skills_from_text(cleaned_job)
    job_embedding = embedding_engine.encode_text(cleaned_job)

    resume_payload = []
    for uploaded_file in uploaded_files:
        resume_text = extract_text_from_pdf(uploaded_file)
        cleaned_resume = clean_resume_text(resume_text)
        resume_skills = skill_extractor.extract_skills_from_text(cleaned_resume)
        resume_payload.append(
            {
                "file_name": uploaded_file.name,
                "resume_text": cleaned_resume,
                "resume_skills": resume_skills,
            }
        )

    resume_embeddings = embedding_engine.encode_batch(
        [item["resume_text"] for item in resume_payload]
    )
    vector_store.build_index(resume_embeddings, resume_payload)

    nearest = vector_store.search(job_embedding, top_k=len(resume_payload))

    ranked_results = []
    for item in nearest:
        match_result = matcher.calculate_comprehensive_match(
            resume_skills=item["resume_skills"],
            job_skills=job_skills,
            resume_text=item["resume_text"],
            job_text=cleaned_job,
        )
        ranked_results.append(
            {
                "file_name": item["file_name"],
                "vector_similarity": item["vector_similarity"],
                **match_result,
                "resume_skills": item["resume_skills"],
            }
        )

    ranked_results = sorted(ranked_results, key=lambda x: x["match_percentage"], reverse=True)

    llm_feedback = ""
    if ranked_results:
        top = ranked_results[0]
        llm_feedback = feedback_generator.generate_improvement_suggestions(
            resume_skills=top["resume_skills"],
            job_skills=job_skills,
            skill_gaps=top["skill_gaps"],
            match_percentage=top["match_percentage"],
            structured_context={
                "resume_skills": top["resume_skills"],
                "job_skills": job_skills,
                "common_skills": top["common_skills"],
                "skill_gaps": top["skill_gaps"],
                "semantic_similarity": top["semantic_similarity"],
                "skill_overlap_score": top["skill_overlap_score"],
                "experience_match_score": top["experience_match_score"],
                "match_percentage": top["match_percentage"],
            },
        )

    st.session_state.ranked_results = ranked_results
    st.session_state.job_skills = job_skills
    st.session_state.llm_feedback = llm_feedback
    st.session_state.analysis_complete = True


def display_results():
    ranked_results = st.session_state.ranked_results

    st.subheader("📊 Ranked Resume Results")
    if not ranked_results:
        st.warning("No ranked results found.")
        return

    top_resume = ranked_results[0]
    metric_cols = st.columns(4)
    metric_cols[0].metric("🏆 Top Resume", top_resume["file_name"])
    metric_cols[1].metric("📈 Top Match", f"{top_resume['match_percentage']:.2f}%")
    metric_cols[2].metric("🧠 Semantic", f"{top_resume['semantic_similarity']:.2f}")
    metric_cols[3].metric("🎯 Skill Overlap", f"{top_resume['skill_overlap_score']:.2f}")

    ranking_rows = []
    for idx, result in enumerate(ranked_results, start=1):
        ranking_rows.append(
            {
                "Rank": idx,
                "Resume": result["file_name"],
                "Final Score (%)": round(result["match_percentage"], 2),
                "Semantic": round(result["semantic_similarity"], 3),
                "Skill Overlap": round(result["skill_overlap_score"], 3),
                "Experience Match": round(result["experience_match_score"], 3),
                "Matched Skills": result["matched_skills_count"],
                "Missing Skills": result["missing_skills_count"],
            }
        )

    st.dataframe(ranking_rows, use_container_width=True)

    st.markdown("### 🧩 Detailed Breakdown")
    for idx, result in enumerate(ranked_results, start=1):
        with st.expander(f"#{idx} - {result['file_name']} ({result['match_percentage']:.2f}%)"):
            st.write("**Common Skills:**", ", ".join(result["common_skills"]) or "None")
            st.write("**Missing Skills:**", ", ".join(result["skill_gaps"]) or "None")
            st.write(
                "**Experience:**",
                f"Resume {result['resume_experience_years']} yrs vs Required {result['required_experience_years']} yrs",
            )

    st.markdown("### 💡 RAG-based LLM Feedback (Top Resume)")
    if st.session_state.llm_feedback:
        st.info(st.session_state.llm_feedback)


def main():
    st.set_page_config(page_title="AI Resume Analyzer & Job Matcher", page_icon="📄", layout="wide")
    st.title("🤖 AI Resume Analyzer & Job Matcher")
    st.caption("Embedding + FAISS + Hybrid Scoring + RAG Feedback")

    with st.sidebar:
        st.header("⚙️ Settings")
        model_option = st.selectbox("Select LLM Model", ["phi3", "llama3"])
        st.markdown("---")
        st.markdown("**Pipeline:** Preprocess → Embedding → FAISS → Hybrid Score → RAG Feedback")

    uploaded_files = st.file_uploader(
        "Upload one or more resumes (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
    )
    job_description = st.text_area(
        "Paste Job Description",
        height=220,
        placeholder="Paste the target job description here...",
    )

    if st.button("🔍 Rank Resumes", type="primary", disabled=not (uploaded_files and job_description)):
        with st.spinner("Analyzing and ranking resumes..."):
            try:
                run_analysis(uploaded_files, job_description, model_option)
                st.success("Analysis complete.")
            except Exception as exc:
                st.error(f"Error during analysis: {exc}")

    if st.session_state.analysis_complete:
        display_results()


if __name__ == "__main__":
    main()
