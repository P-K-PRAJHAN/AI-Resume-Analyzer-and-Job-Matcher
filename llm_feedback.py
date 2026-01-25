import ollama
from typing import List, Dict


class LLMFeedbackGenerator:
    def __init__(self, model_name='phi3'):
        """
        Initialize the LLM feedback generator

        Args:
            model_name (str): Name of the Ollama model to use (default: phi3)
        """
        self.model_name = model_name

    def generate_improvement_suggestions(self, resume_skills: List[str], job_skills: List[str],
                                         skill_gaps: List[str], match_percentage: float) -> str:
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
        if not skill_gaps:
            prompt = f"""
            The candidate's resume matches the job requirements very well with a {match_percentage}% match.
            The candidate has the following skills: {', '.join(resume_skills[:10])}.
            The job requires: {', '.join(job_skills[:10])}.
            
            Provide positive feedback and suggest how the candidate can further enhance their profile for similar roles.
            Keep the response concise and encouraging.
            """
        else:
            prompt = f"""
            Analyze the resume against the job description and provide improvement suggestions.
            
            Resume Skills: {', '.join(resume_skills)}
            Job Requirements: {', '.join(job_skills)}
            Missing Skills: {', '.join(skill_gaps)}
            Current Match: {match_percentage}%
            
            Provide specific, actionable improvement suggestions to help the candidate improve their match score.
            Organize suggestions by:
            1. Top priority skills to learn
            2. Recommended learning resources or courses
            3. How to showcase existing skills better
            4. General tips to improve the resume
            
            Be specific, constructive, and encouraging. Limit response to 200 words.
            """

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ]
            )
            return response['message']['content']
        except Exception as e:
            return f"Error generating feedback: {str(e)}. Please ensure Ollama is installed and running with the '{self.model_name}' model."

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
        Generate suggestions for customizing a cover letter based on the resume and job description.
        
        Resume Skills: {', '.join(resume_skills)}
        Job Requirements: {', '.join(job_skills)}
        Job Description: {job_description[:500]}  # Limit length
        
        Provide specific advice on:
        1. Which resume skills to highlight
        2. How to align experience with job requirements
        3. Key phrases to include
        4. Structure recommendations
        
        Keep suggestions concise and actionable.
        """

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ]
            )
            return response['message']['content']
        except Exception as e:
            return f"Error generating cover letter suggestions: {str(e)}"

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

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ]
            )
            return response['message']['content']
        except Exception as e:
            return f"Error generating learning path: {str(e)}"


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
