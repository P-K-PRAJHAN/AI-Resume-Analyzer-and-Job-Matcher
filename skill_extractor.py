import spacy
from spacy.matcher import PhraseMatcher
from typing import List, Dict


class SkillExtractor:
    def __init__(self):
        """
        Initialize the skill extractor with spaCy model and predefined skill sets
        """
        try:
            # Load the English language model
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("spaCy model 'en_core_web_sm' not found. Install it using: python -m spacy download en_core_web_sm")
            raise

        self.tech_skills = {
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust',
            'sql', 'html', 'css', 'react', 'angular', 'vue', 'django', 'flask',
            'fastapi', 'spring', 'node.js', 'express', 'tensorflow', 'pytorch', 'keras',
            'pandas', 'numpy', 'scikit-learn', 'opencv', 'spark', 'hadoop', 'graphql',
            'rest', 'docker', 'kubernetes', 'aws', 'azure', 'gcp', 'git', 'github',
            'jenkins', 'terraform', 'linux', 'mysql', 'postgresql', 'mongodb', 'redis',
            'machine learning', 'artificial intelligence', 'nlp', 'data science',
            'deep learning', 'llm', 'langchain', 'rag', 'faiss', 'streamlit', 'ci/cd'
        }

        # Soft skills
        self.soft_skills = {
            'communication', 'leadership', 'teamwork', 'problem solving', 'critical thinking',
            'creativity', 'adaptability', 'time management', 'collaboration', 'interpersonal',
            'negotiation', 'presentation', 'empathy', 'conflict resolution', 'decision making',
            'organizational', 'analytical', 'attention to detail', 'multitasking', 'stress management'
        }

        self.all_skills = self.tech_skills.union(self.soft_skills)

        self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        patterns = [self.nlp.make_doc(skill) for skill in sorted(self.all_skills)]
        self.matcher.add("SKILLS", patterns)

        self.skill_aliases = {
            "nodejs": "node.js",
            "node js": "node.js",
            "js": "javascript",
            "ml": "machine learning",
            "ai": "artificial intelligence",
            "nlp": "nlp",
            "ci cd": "ci/cd",
        }

    def _normalize_skill(self, skill: str) -> str:
        normalized = skill.lower().strip()
        return self.skill_aliases.get(normalized, normalized)

    def extract_skills_from_text(self, text: str) -> List[str]:
        """
        Extract skills from the given text using spaCy and pattern matching

        Args:
            text (str): Input text to extract skills from

        Returns:
            List[str]: List of extracted skills
        """
        if not text:
            return []

        doc = self.nlp(text.lower())
        found_skills = set()

        for match_id, start, end in self.matcher(doc):
            found_skills.add(self._normalize_skill(doc[start:end].text))

        lowered_text = text.lower()
        for skill in self.all_skills:
            if skill in lowered_text:
                found_skills.add(self._normalize_skill(skill))

        return sorted([skill for skill in found_skills if skill in self.all_skills])

    def extract_technical_skills(self, text: str) -> List[str]:
        """
        Extract only technical skills from the given text

        Args:
            text (str): Input text to extract skills from

        Returns:
            List[str]: List of extracted technical skills
        """
        all_skills = self.extract_skills_from_text(text)
        tech_skills = []

        for skill in all_skills:
            if skill in self.tech_skills:
                tech_skills.append(skill)

        return tech_skills

    def extract_soft_skills(self, text: str) -> List[str]:
        """
        Extract only soft skills from the given text

        Args:
            text (str): Input text to extract skills from

        Returns:
            List[str]: List of extracted soft skills
        """
        all_skills = self.extract_skills_from_text(text)
        soft_skills = []

        for skill in all_skills:
            if skill in self.soft_skills:
                soft_skills.append(skill)

        return soft_skills

    def get_skill_categories(self, skills: List[str]) -> dict:
        """
        Categorize skills into technical and soft skills

        Args:
            skills (List[str]): List of skills to categorize

        Returns:
            dict: Dictionary with 'technical' and 'soft' keys
        """
        technical = []
        soft = []

        for skill in skills:
            if skill in self.tech_skills:
                technical.append(skill)
            elif skill in self.soft_skills:
                soft.append(skill)

        return {
            'technical': technical,
            'soft': soft
        }

    def compute_skill_overlap(self, resume_skills: List[str], job_skills: List[str]) -> Dict[str, object]:
        resume_set = {self._normalize_skill(skill) for skill in resume_skills}
        job_set = {self._normalize_skill(skill) for skill in job_skills}

        if not job_set:
            return {
                "overlap_score": 0.0,
                "common_skills": [],
                "missing_skills": [],
                "matched_count": 0,
                "required_count": 0,
            }

        common = sorted(resume_set.intersection(job_set))
        missing = sorted(job_set - resume_set)
        overlap_score = len(common) / len(job_set)

        return {
            "overlap_score": overlap_score,
            "common_skills": common,
            "missing_skills": missing,
            "matched_count": len(common),
            "required_count": len(job_set),
        }


# Example usage
if __name__ == "__main__":
    extractor = SkillExtractor()

    sample_text = """
    Experienced software engineer with 5 years of experience in Python, Java, and JavaScript.
    Proficient in Django, React, and AWS cloud technologies. Strong problem solving and 
    communication skills. Familiar with machine learning, data science, and CI/CD pipelines.
    """

    skills = extractor.extract_skills_from_text(sample_text)
    print("Extracted Skills:", skills)

    tech_skills = extractor.extract_technical_skills(sample_text)
    print("Technical Skills:", tech_skills)

    soft_skills = extractor.extract_soft_skills(sample_text)
    print("Soft Skills:", soft_skills)
