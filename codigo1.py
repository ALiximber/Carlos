import pdfplumber
import pytesseract
from pdf2image import convert_from_path
import docx
import os

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    
    if not text.strip():
        images = convert_from_path(pdf_path)
        for img in images:
            text += pytesseract.image_to_string(img)
    
    return text.strip()

def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_all_cvs(folder_path="cv_data"):
    cv_texts = []
    
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if file.endswith(".pdf"):
            cv_texts.append(extract_text_from_pdf(file_path))
        elif file.endswith(".docx"):
            cv_texts.append(extract_text_from_docx(file_path))
    
    return cv_texts

cv_texts = extract_all_cvs()
print("Se extrajeron CVs:", len(cv_texts))
