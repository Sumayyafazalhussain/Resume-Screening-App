
from io import BytesIO
from docx import Document
from pypdf import PdfReader

def read_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(file_bytes))
    texts = []
    for p in reader.pages:
        texts.append(p.extract_text() or "")
    return "\\n".join(texts)

def read_docx(file_bytes: bytes) -> str:
    bio = BytesIO(file_bytes)
    doc = Document(bio)
    return "\\n".join([p.text for p in doc.paragraphs])

def read_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore")

def extract_text(filename: str, file_bytes: bytes) -> str:
    fn = filename.lower()
    if fn.endswith(".pdf"):
        return read_pdf(file_bytes)
    if fn.endswith(".docx"):
        return read_docx(file_bytes)
    if fn.endswith(".txt"):
        return read_txt(file_bytes)
    raise ValueError("Unsupported file type")
