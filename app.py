import streamlit as st
from resume_parser import extract_text_from_pdf, clean_resume_text
from skill_extractor import SkillExtractor
from matcher import ResumeJobMatcher
from llm_feedback import LLMFeedbackGenerator
import os


# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'resume_skills' not in st.session_state:
    st.session_state.resume_skills = []
if 'job_skills' not in st.session_state:
    st.session_state.job_skills = []


def main():
    st.set_page_config(
        page_title="AI Resume Analyzer & Job Matcher",
        page_icon="📄",
        layout="wide"
    )

    st.title("🤖 AI Resume Analyzer & Job Matcher")
    st.markdown("""
    Analyze your resume against job descriptions and get personalized improvement suggestions.
    All processing happens locally - your data never leaves your computer!
    """)

    # Sidebar with instructions
    with st.sidebar:
        st.header("📋 How to Use")
        st.markdown("""
        1. Upload your resume (PDF format)
        2. Paste the job description
        3. Click "Analyze Match" to get results
        4. Review skills match and improvement suggestions
        """)

        st.header("⚙️ Settings")
        model_option = st.selectbox(
            "Select LLM Model",
            ["phi3", "llama3"],
            help="Choose the local LLM model for generating suggestions"
        )

        st.markdown("---")
        st.info(
            "💡 Tip: Make sure Ollama is running with your selected model installed.")

    # Main content
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📄 Upload Resume")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload your resume in PDF format"
        )

        if uploaded_file is not None:
            st.success("✅ Resume uploaded successfully!")

    with col2:
        st.subheader("💼 Job Description")
        job_description = st.text_area(
            "Paste the job description here",
            height=200,
            placeholder="Enter the job description you want to match your resume against..."
        )

        if job_description:
            st.success("✅ Job description entered!")

    # Analysis button
    analyze_button = st.button(
        "🔍 Analyze Match",
        type="primary",
        disabled=not (uploaded_file and job_description)
    )

    # Perform analysis
    if analyze_button:
        with st.spinner("🔄 Analyzing your resume against the job description..."):
            try:
                # Initialize components
                skill_extractor = SkillExtractor()
                matcher = ResumeJobMatcher()
                feedback_generator = LLMFeedbackGenerator(
                    model_name=model_option)

                # Extract text from PDF
                resume_text = extract_text_from_pdf(uploaded_file)
                cleaned_resume = clean_resume_text(resume_text)

                # Extract skills
                resume_skills = skill_extractor.extract_skills_from_text(
                    cleaned_resume)
                job_skills = skill_extractor.extract_skills_from_text(
                    job_description)

                # Store in session state
                st.session_state.resume_skills = resume_skills
                st.session_state.job_skills = job_skills

                # Calculate match
                results = matcher.calculate_comprehensive_match(
                    resume_skills, job_skills)

                # Generate feedback
                improvement_suggestions = feedback_generator.generate_improvement_suggestions(
                    resume_skills,
                    job_skills,
                    results['skill_gaps'],
                    results['match_percentage']
                )

                # Store results and components in session state
                st.session_state.results = {
                    'match_results': results,
                    'improvement_suggestions': improvement_suggestions,
                    'resume_text': cleaned_resume,
                    'job_description': job_description
                }
                st.session_state.feedback_generator = feedback_generator
                st.session_state.analysis_complete = True

                st.success("✅ Analysis complete!")

            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")

    # Display results if analysis is complete
    if st.session_state.analysis_complete and st.session_state.results:
        display_results()


def display_results():
    results = st.session_state.results
    match_results = results['match_results']

    st.markdown("---")
    st.subheader("📊 Analysis Results")

    # Display match percentage in a metric card
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="🎯 Match Score",
            value=f"{match_results['match_percentage']}%",
            delta=f"{match_results['matched_skills_count']}/{match_results['total_job_skills']} skills matched"
        )

    with col2:
        st.metric(
            label="✅ Matched Skills",
            value=match_results['matched_skills_count']
        )

    with col3:
        st.metric(
            label="❌ Missing Skills",
            value=match_results['missing_skills_count']
        )

    with col4:
        st.metric(
            label="📝 Total Job Skills",
            value=match_results['total_job_skills']
        )

    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Skills Overview",
        "🔍 Detailed Match",
        "💡 Improvement Suggestions",
        "📈 Skill Gap Analysis"
    ])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📌 Your Skills")
            if st.session_state.resume_skills:
                resume_skills_str = ", ".join(st.session_state.resume_skills)
                st.text_area("", value=resume_skills_str,
                             height=200, disabled=True)
            else:
                st.warning("No skills detected in resume")

        with col2:
            st.markdown("### 🎯 Required Skills")
            if st.session_state.job_skills:
                job_skills_str = ", ".join(st.session_state.job_skills)
                st.text_area("", value=job_skills_str,
                             height=200, disabled=True)
            else:
                st.warning("No skills detected in job description")

    with tab2:
        st.markdown("### 📊 Match Details")

        # Show common skills
        st.markdown("#### ✅ Skills You Have That Match")
        if match_results['common_skills']:
            common_skills_str = ", ".join(match_results['common_skills'])
            st.text_area("", value=common_skills_str,
                         height=100, disabled=True)
        else:
            st.info("No matching skills found")

        # Show match statistics
        st.markdown("#### 📈 Matching Statistics")
        stats_col1, stats_col2 = st.columns(2)

        with stats_col1:
            st.write("**Semantic Similarity:**",
                     f"{match_results['semantic_similarity']:.2%}")
            st.write("**Keyword Match:**",
                     f"{match_results['keyword_match_percentage']:.1f}%")

        with stats_col2:
            st.write("**Weighted Score:**",
                     f"{match_results['weighted_score']:.2%}")
            st.write("**Match Quality:**",
                     "Excellent" if match_results['match_percentage'] >= 80
                     else "Good" if match_results['match_percentage'] >= 60
                     else "Fair" if match_results['match_percentage'] >= 40
                     else "Needs Improvement")

    with tab3:
        st.markdown("### 💡 Personalized Improvement Suggestions")

        if results['improvement_suggestions']:
            st.markdown(f"<div style='background-color: #262730; color: white; padding: 20px; border-radius: 10px; font-family: sans-serif;'>{results['improvement_suggestions']}</div>",
                        unsafe_allow_html=True)
        else:
            st.warning(
                "Could not generate improvement suggestions. Please ensure Ollama is running.")

    with tab4:
        st.markdown("### ❌ Missing Skills (Skill Gaps)")

        if match_results['skill_gaps']:
            # Create a dataframe-like display for missing skills
            missing_skills_str = ", ".join(match_results['skill_gaps'])
            st.text_area("", value=missing_skills_str,
                         height=150, disabled=True)

            st.markdown("#### 📚 Recommended Learning Path")
            with st.spinner("Generating learning recommendations..."):
                try:
                    # Use the feedback generator from session state
                    learning_path = st.session_state.feedback_generator.generate_skill_learning_path(
                        match_results['skill_gaps'])

                    if learning_path and "Error" not in learning_path:
                        st.markdown(f"<div style='background-color: #1e3a5f; color: white; padding: 15px; border-radius: 8px; border-left: 4px solid #4da6ff; font-family: sans-serif;'>{learning_path}</div>",
                                    unsafe_allow_html=True)
                    else:
                        st.warning(
                            "Could not generate learning path. Please check Ollama connection.")
                except Exception as e:
                    st.error(f"Error generating learning path: {str(e)}")
        else:
            st.success(
                "🎉 Congratulations! You have all the required skills for this position.")


if __name__ == "__main__":
    main()
