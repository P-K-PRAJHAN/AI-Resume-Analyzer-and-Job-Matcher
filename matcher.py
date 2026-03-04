from typing import Dict, List

import numpy as np

from embedding_engine import EmbeddingEngine
from scoring_engine import ScoringEngine
from skill_extractor import SkillExtractor


class ResumeJobMatcher:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the matcher with a sentence transformer model

        Args:
            model_name (str): Name of the sentence transformer model to use
        """
        self.embedding_engine = EmbeddingEngine(model_name=model_name)
        self.skill_extractor = SkillExtractor()
        self.scoring_engine = ScoringEngine()

    def calculate_similarity(self, resume_text: str, job_text: str) -> float:
        """
        Calculate the semantic similarity between resume skills and job skills

        Args:
            resume_skills (List[str]): List of skills from resume
            job_skills (List[str]): List of skills from job description

        Returns:
            float: Similarity score between 0 and 1 (as percentage when multiplied by 100)
        """
        if not resume_text or not job_text:
            return 0.0

        resume_embedding = self.embedding_engine.encode_text(resume_text)
        job_embedding = self.embedding_engine.encode_text(job_text)
        similarity_score = float(np.dot(resume_embedding, job_embedding))
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
        overlap = self.skill_extractor.compute_skill_overlap(resume_skills, job_skills)
        return overlap["overlap_score"] * 100

    def get_skill_gaps(self, resume_skills: List[str], job_skills: List[str]) -> List[str]:
        """
        Identify skills present in job description but missing from resume

        Args:
            resume_skills (List[str]): List of skills from resume
            job_skills (List[str]): List of skills from job description

        Returns:
            List[str]: List of missing skills
        """
        overlap = self.skill_extractor.compute_skill_overlap(resume_skills, job_skills)
        return overlap["missing_skills"]

    def get_common_skills(self, resume_skills: List[str], job_skills: List[str]) -> List[str]:
        """
        Identify skills present in both resume and job description

        Args:
            resume_skills (List[str]): List of skills from resume
            job_skills (List[str]): List of skills from job description

        Returns:
            List[str]: List of common skills
        """
        overlap = self.skill_extractor.compute_skill_overlap(resume_skills, job_skills)
        return overlap["common_skills"]

    def calculate_comprehensive_match(
        self,
        resume_skills: List[str],
        job_skills: List[str],
        resume_text: str = "",
        job_text: str = "",
    ) -> Dict[str, object]:
        """
        Calculate comprehensive match metrics between resume and job

        Args:
            resume_skills (List[str]): List of skills from resume
            job_skills (List[str]): List of skills from job description

        Returns:
            dict: Dictionary containing various match metrics
        """
        semantic_similarity = self.calculate_similarity(resume_text, job_text)
        overlap = self.skill_extractor.compute_skill_overlap(resume_skills, job_skills)
        experience_info = self.scoring_engine.compute_experience_match(resume_text, job_text)
        final_score = self.scoring_engine.compute_final_score(
            semantic_similarity=semantic_similarity,
            skill_overlap=overlap["overlap_score"],
            experience_match=experience_info["experience_match"],
        )

        skill_gaps = overlap["missing_skills"]
        common_skills = overlap["common_skills"]
        keyword_match_percentage = overlap["overlap_score"] * 100

        return {
            'semantic_similarity': semantic_similarity,
            'keyword_match_percentage': keyword_match_percentage,
            'skill_overlap_score': overlap["overlap_score"],
            'experience_match_score': experience_info["experience_match"],
            'resume_experience_years': experience_info["resume_years"],
            'required_experience_years': experience_info["required_years"],
            'weighted_score': final_score,
            'match_percentage': round(final_score * 100, 2),
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
