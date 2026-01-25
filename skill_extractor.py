import spacy
import re
from typing import List, Set


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

        # Define common skill patterns and keywords
        self.tech_skills = {
            # Programming languages
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'c', 'ruby', 'php', 'go', 'rust',
            'swift', 'kotlin', 'scala', 'perl', 'r', 'matlab', 'sql', 'html', 'css', 'dart',

            # Frameworks and Libraries
            'react', 'angular', 'vue', 'django', 'flask', 'spring', 'node.js', 'express', 'laravel',
            'tensorflow', 'pytorch', 'keras', 'pandas', 'numpy', 'scikit-learn', 'opencv', 'spark',
            'hadoop', 'redux', 'graphql', 'rest', 'api', 'docker', 'kubernetes', 'aws', 'azure', 'gcp',

            # Technologies and Tools
            'git', 'github', 'gitlab', 'jenkins', 'ansible', 'terraform', 'linux', 'unix', 'windows',
            'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'oracle', 'sqlite',
            'machine learning', 'artificial intelligence', 'nlp', 'computer vision', 'data science',
            'big data', 'cloud computing', 'devops', 'agile', 'scrum', 'ci/cd'
        }

        # Soft skills
        self.soft_skills = {
            'communication', 'leadership', 'teamwork', 'problem solving', 'critical thinking',
            'creativity', 'adaptability', 'time management', 'collaboration', 'interpersonal',
            'negotiation', 'presentation', 'empathy', 'conflict resolution', 'decision making',
            'organizational', 'analytical', 'attention to detail', 'multitasking', 'stress management'
        }

        # Combine all skills
        self.all_skills = self.tech_skills.union(self.soft_skills)

    def extract_skills_from_text(self, text: str) -> List[str]:
        """
        Extract skills from the given text using spaCy and pattern matching

        Args:
            text (str): Input text to extract skills from

        Returns:
            List[str]: List of extracted skills
        """
        # Process the text with spaCy
        doc = self.nlp(text.lower())

        found_skills = set()

        # Method 1: Direct keyword matching
        for skill in self.all_skills:
            if skill in text.lower():
                found_skills.add(skill)

        # Method 2: Entity recognition for relevant skills
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PRODUCT', 'WORK_OF_ART', 'EVENT', 'LAW']:
                # Check if entity matches any known skills
                entity_text = ent.text.lower().strip()
                for skill in self.all_skills:
                    if skill in entity_text or entity_text in skill:
                        found_skills.add(skill)

        # Method 3: Pattern-based extraction
        # Look for phrases that might indicate skills (e.g., "experienced in X", "proficient with Y")
        patterns = [
            r'experienced in ([a-zA-Z\s]+)',
            r'proficient in ([a-zA-Z\s]+)',
            r'knowledge of ([a-zA-Z\s]+)',
            r'familiar with ([a-zA-Z\s]+)',
            r'skilled in ([a-zA-Z\s]+)'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                skill_candidate = match.strip()
                for skill in self.all_skills:
                    if skill in skill_candidate or skill_candidate in skill:
                        found_skills.add(skill)

        return sorted(list(found_skills))

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
