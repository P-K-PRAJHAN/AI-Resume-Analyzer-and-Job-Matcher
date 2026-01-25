from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
from typing import List, Tuple


class ResumeJobMatcher:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the matcher with a sentence transformer model

        Args:
            model_name (str): Name of the sentence transformer model to use
        """
        self.model = SentenceTransformer(model_name)

    def calculate_similarity(self, resume_skills: List[str], job_skills: List[str]) -> float:
        """
        Calculate the semantic similarity between resume skills and job skills

        Args:
            resume_skills (List[str]): List of skills from resume
            job_skills (List[str]): List of skills from job description

        Returns:
            float: Similarity score between 0 and 1 (as percentage when multiplied by 100)
        """
        if not resume_skills or not job_skills:
            return 0.0

        # Convert lists to sentences for embedding
        resume_text = " ".join(resume_skills)
        job_text = " ".join(job_skills)

        # Generate embeddings
        resume_embedding = self.model.encode(
            resume_text, convert_to_tensor=True)
        job_embedding = self.model.encode(job_text, convert_to_tensor=True)

        # Calculate cosine similarity
        cosine_score = util.cos_sim(resume_embedding, job_embedding)

        # Convert tensor to float and return
        similarity_score = cosine_score.item()

        # Ensure the score is between 0 and 1
        return max(0.0, min(1.0, similarity_score))

    def calculate_keyword_match_percentage(self, resume_skills: List[str], job_skills: List[str]) -> float:
        """
        Calculate the percentage of job skills that appear in resume skills

        Args:
            resume_skills (List[str]): List of skills from resume
            job_skills (List[str]): List of skills from job description

        Returns:
            float: Percentage of matched skills (0-100)
        """
        if not job_skills:
            return 0.0

        resume_set = set(skill.lower().strip() for skill in resume_skills)
        job_set = set(skill.lower().strip() for skill in job_skills)

        # Calculate intersection
        matched_skills = resume_set.intersection(job_set)

        # Calculate percentage
        match_percentage = (len(matched_skills) /
                            len(job_set)) * 100 if job_set else 0.0

        return match_percentage

    def get_skill_gaps(self, resume_skills: List[str], job_skills: List[str]) -> List[str]:
        """
        Identify skills present in job description but missing from resume

        Args:
            resume_skills (List[str]): List of skills from resume
            job_skills (List[str]): List of skills from job description

        Returns:
            List[str]: List of missing skills
        """
        resume_set = set(skill.lower().strip() for skill in resume_skills)
        job_set = set(skill.lower().strip() for skill in job_skills)

        # Find skills in job but not in resume
        missing_skills = job_set - resume_set

        return sorted(list(missing_skills))

    def get_common_skills(self, resume_skills: List[str], job_skills: List[str]) -> List[str]:
        """
        Identify skills present in both resume and job description

        Args:
            resume_skills (List[str]): List of skills from resume
            job_skills (List[str]): List of skills from job description

        Returns:
            List[str]: List of common skills
        """
        resume_set = set(skill.lower().strip() for skill in resume_skills)
        job_set = set(skill.lower().strip() for skill in job_skills)

        # Find common skills
        common_skills = resume_set.intersection(job_set)

        return sorted(list(common_skills))

    def calculate_comprehensive_match(self, resume_skills: List[str], job_skills: List[str]) -> dict:
        """
        Calculate comprehensive match metrics between resume and job

        Args:
            resume_skills (List[str]): List of skills from resume
            job_skills (List[str]): List of skills from job description

        Returns:
            dict: Dictionary containing various match metrics
        """
        semantic_similarity = self.calculate_similarity(
            resume_skills, job_skills)
        keyword_match_percentage = self.calculate_keyword_match_percentage(
            resume_skills, job_skills)
        skill_gaps = self.get_skill_gaps(resume_skills, job_skills)
        common_skills = self.get_common_skills(resume_skills, job_skills)

        # Weighted score combining semantic similarity and keyword match
        # Give 70% weight to semantic similarity and 30% to keyword match
        weighted_score = (semantic_similarity * 0.7) + \
            ((keyword_match_percentage / 100) * 0.3)

        return {
            'semantic_similarity': semantic_similarity,
            'keyword_match_percentage': keyword_match_percentage,
            'weighted_score': weighted_score,
            'match_percentage': round(weighted_score * 100, 2),
            'skill_gaps': skill_gaps,
            'common_skills': common_skills,
            'total_job_skills': len(job_skills),
            'matched_skills_count': len(common_skills),
            'missing_skills_count': len(skill_gaps)
        }


# Example usage
if __name__ == "__main__":
    matcher = ResumeJobMatcher()

    resume_skills = [
        "python", "machine learning", "data analysis", "sql", "pandas",
        "numpy", "communication", "problem solving"
    ]

    job_skills = [
        "python", "machine learning", "deep learning", "tensorflow", "pytorch",
        "data science", "sql", "communication", "teamwork", "project management"
    ]

    results = matcher.calculate_comprehensive_match(resume_skills, job_skills)

    print("Match Results:")
    print(f"Match Percentage: {results['match_percentage']}%")
    print(f"Common Skills: {results['common_skills']}")
    print(f"Skill Gaps: {results['skill_gaps']}")
    print(f"Missing Skills Count: {results['missing_skills_count']}")
    print(f"Matched Skills Count: {results['matched_skills_count']}")
