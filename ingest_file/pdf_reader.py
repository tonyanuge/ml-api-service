import pdfplumber
from io import BytesIO
from pdf2image import convert_from_bytes
import pytesseract

# Set this if Tesseract isn't in PATH:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
import os
os.environ["PATH"] += os.pathsep + os.environ.get("POPPLER_PATH", "")

POPPLER_PATH = r"C:\Users\t_all\poppler-25.12.0\Library\bin"  # update to your actual path


def extract_pdf_text(file_bytes: bytes) -> str:
    """
    Extract text from PDF.
    1. Try pdfplumber (works for text-based PDFs)
    2. If empty → fallback to OCR for scanned PDFs
    """
    text = ""

    # ---------------------------------------------------
    # 1. Try text extraction (normal PDFs)
    # ---------------------------------------------------
    try:
        pdf_stream = BytesIO(file_bytes)
        with pdfplumber.open(pdf_stream) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"

        if text.strip():
            print("[PDF] Extracted text using pdfplumber")
            return text.strip()

    except Exception as e:
        print("[PDF] pdfplumber failed → switching to OCR:", e)

    # ---------------------------------------------------
    # 2. OCR fallback (scanned PDFs)
    # ---------------------------------------------------
    try:
        images = convert_from_bytes(file_bytes, poppler_path=POPPLER_PATH)
        ocr_text = ""

        for img in images:
            page_text = pytesseract.image_to_string(img)
            ocr_text += page_text + "\n"

        print("[PDF OCR] Extracted text using Tesseract OCR")
        return ocr_text.strip()

    except Exception as e:
        print("[OCR ERROR]", e)
        return ""
