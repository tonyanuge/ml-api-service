from docx import Document
from io import BytesIO

def extract_docx_text(file_bytes: bytes) -> str:
    doc = Document(BytesIO(file_bytes))
    return "\n".join([p.text for p in doc.paragraphs])
