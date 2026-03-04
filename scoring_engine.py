import re
from typing import Dict


class ScoringEngine:
    def __init__(self, semantic_weight: float = 0.5, skill_weight: float = 0.3, experience_weight: float = 0.2):
        self.semantic_weight = semantic_weight
        self.skill_weight = skill_weight
        self.experience_weight = experience_weight

    def extract_experience_years(self, text: str) -> float:
        if not text:
            return 0.0
        matches = re.findall(r"(\d+(?:\.\d+)?)\+?\s*(?:years|yrs)", text.lower())
        if not matches:
            return 0.0
        return max(float(value) for value in matches)

    def compute_experience_match(self, resume_text: str, job_description: str) -> Dict[str, float]:
        resume_years = self.extract_experience_years(resume_text)
        required_years = self.extract_experience_years(job_description)

        if required_years == 0:
            return {
                "resume_years": resume_years,
                "required_years": 0.0,
                "experience_match": 1.0,
            }

        ratio = min(resume_years / required_years, 1.0)
        return {
            "resume_years": resume_years,
            "required_years": required_years,
            "experience_match": ratio,
        }

    def compute_final_score(self, semantic_similarity: float, skill_overlap: float, experience_match: float) -> float:
        score = (
            (self.semantic_weight * semantic_similarity)
            + (self.skill_weight * skill_overlap)
            + (self.experience_weight * experience_match)
        )
        return max(0.0, min(1.0, score))
