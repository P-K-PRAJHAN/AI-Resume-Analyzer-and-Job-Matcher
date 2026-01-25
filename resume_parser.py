import PyPDF2
from io import BytesIO


def extract_text_from_pdf(pdf_file):
    """
    Extract text content from a PDF file

    Args:
        pdf_file: Uploaded file object from Streamlit

    Returns:
        str: Extracted text from the PDF
    """
    try:
        # Read the PDF file
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""

        # Extract text from each page
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[0]
            text += page.extract_text()

        return text.strip()
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
    # Remove extra whitespaces and normalize
    cleaned_text = ' '.join(text.split())
    return cleaned_text
