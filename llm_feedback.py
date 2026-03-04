import ollama
from typing import Dict, List


class LLMFeedbackGenerator:
    def __init__(self, model_name='phi3'):
        """
        Initialize the LLM feedback generator

        Args:
            model_name (str): Name of the Ollama model to use (default: phi3)
        """
        self.model_name = model_name

    def _build_ollama_connection_error(self, original_error: Exception) -> str:
        return (
            "Failed to connect to Ollama server. "
            "Start Ollama first using `ollama serve`, then retry. "
            f"Original error: {original_error}"
        )

    def _chat(self, prompt: str) -> str:
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}]
            )
            return response['message']['content']
        except Exception as exc:
            error_text = str(exc).lower()
            if "failed to connect" in error_text or "connection" in error_text:
                return self._build_ollama_connection_error(exc)
            if "not found" in error_text or "pull" in error_text:
                return (
                    f"Model '{self.model_name}' was not found locally. "
                    f"Install it using: `ollama pull {self.model_name}`. "
                    f"Original error: {exc}"
                )
            return f"Ollama request failed: {exc}"

    def _build_rag_prompt(self, context: Dict[str, object]) -> str:
        return f"""
You are an AI resume coach.

Use ONLY the provided structured context to produce grounded feedback.
Do not invent information and do not assume details outside this context.

Structured Context:
- Candidate Skills: {', '.join(context.get('resume_skills', []))}
- Job Skills: {', '.join(context.get('job_skills', []))}
- Matched Skills: {', '.join(context.get('common_skills', []))}
- Missing Skills: {', '.join(context.get('skill_gaps', []))}
- Semantic Similarity: {context.get('semantic_similarity', 0):.2f}
- Skill Overlap: {context.get('skill_overlap_score', 0):.2f}
- Experience Match: {context.get('experience_match_score', 0):.2f}
- Final Match Percentage: {context.get('match_percentage', 0):.2f}%

Output format:
1) Overall fit summary (2-3 lines)
2) Top 3 missing priorities
3) Resume rewrite tips (bullet points)
4) 30-day practical action plan

Keep response concise, specific, and encouraging (max 220 words).
""".strip()

    def generate_improvement_suggestions(self, resume_skills: List[str], job_skills: List[str],
                                         skill_gaps: List[str], match_percentage: float,
                                         structured_context: Dict[str, object] | None = None) -> str:
        """
        Generate improvement suggestions using local LLM

        Args:
            resume_skills (List[str]): Skills from resume
            job_skills (List[str]): Skills from job description
            skill_gaps (List[str]): Missing skills identified
            match_percentage (float): Overall match percentage

        Returns:
            str: Improvement suggestions from the LLM
        """
        context = structured_context or {
            "resume_skills": resume_skills,
            "job_skills": job_skills,
            "common_skills": sorted(set(resume_skills).intersection(set(job_skills))),
            "skill_gaps": skill_gaps,
            "semantic_similarity": 0.0,
            "skill_overlap_score": 0.0,
            "experience_match_score": 0.0,
            "match_percentage": match_percentage,
        }
        prompt = self._build_rag_prompt(context)

        return self._chat(prompt)

    def generate_cover_letter_suggestions(self, resume_skills: List[str], job_skills: List[str],
                                          job_description: str) -> str:
        """
        Generate suggestions for customizing a cover letter

        Args:
            resume_skills (List[str]): Skills from resume
            job_skills (List[str]): Skills from job description
            job_description (str): Full job description text

        Returns:
            str: Cover letter suggestions
        """
        prompt = f"""
        Create concise cover-letter guidance using this structured context only:
        - Candidate Skills: {', '.join(resume_skills)}
        - Job Skills: {', '.join(job_skills)}
        - Job Snapshot: {job_description[:350]}

        Provide:
        1) strongest alignment points
        2) 3 quantified impact-style bullet suggestions
        3) closing paragraph guidance
        """

        return self._chat(prompt)

    def generate_skill_learning_path(self, skill_gaps: List[str]) -> str:
        """
        Generate a learning path for missing skills

        Args:
            skill_gaps (List[str]): List of missing skills

        Returns:
            str: Learning path recommendations
        """
        if not skill_gaps:
            return "No additional skills needed! The resume matches well with the job requirements."

        prompt = f"""
        Create a learning path for the following missing skills: {', '.join(skill_gaps[:5])}.
        
        For each skill, provide:
        1. Priority level (High/Medium/Low)
        2. Recommended learning resources (free online courses, tutorials, books)
        3. Estimated time to gain proficiency
        4. Practical projects to demonstrate the skill
        
        Focus on free or low-cost resources. Keep the learning path realistic and achievable.
        """

        return self._chat(prompt)


# Example usage
if __name__ == "__main__":
    feedback_gen = LLMFeedbackGenerator()

    resume_skills = ["python", "machine learning", "data analysis", "sql"]
    job_skills = ["python", "machine learning",
                  "deep learning", "tensorflow", "data science"]
    skill_gaps = ["deep learning", "tensorflow", "data science"]

    suggestions = feedback_gen.generate_improvement_suggestions(
        resume_skills,
        job_skills,
        skill_gaps,
        60.0
    )

    print("Improvement Suggestions:")
    print(suggestions)
