import re
from typing import Iterable

import pdfplumber


def extract_text_from_pdf(pdf_file):
    """
    Extract text content from a PDF file

    Args:
        pdf_file: Uploaded file object from Streamlit

    Returns:
        str: Extracted text from the PDF
    """
    try:
        text_parts = []
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                if page_text:
                    text_parts.append(page_text)
        return "\n".join(text_parts).strip()
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")


def clean_resume_text(text):
    """
    Clean and preprocess the extracted resume text

    Args:
        text (str): Raw text extracted from resume

    Returns:
        str: Cleaned and normalized text
    """
    if not text:
        return ""

    cleaned_text = text
    cleaned_text = cleaned_text.replace("\u00a0", " ")
    cleaned_text = re.sub(r"\s+", " ", cleaned_text)
    cleaned_text = re.sub(r"[^\w\s\-\+\.#/,]", " ", cleaned_text)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    return cleaned_text


def clean_text_batch(texts: Iterable[str]):
    return [clean_resume_text(item) for item in texts]
