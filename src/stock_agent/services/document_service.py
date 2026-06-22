from io import BytesIO

from bs4 import BeautifulSoup
from pypdf import PdfReader


def extract_pdf_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(pdf_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages).strip()


def extract_text(content: bytes, content_type: str = "") -> str:
    lowered = content_type.lower()
    if "pdf" in lowered or content.startswith(b"%PDF"):
        return extract_pdf_text(content)
    decoded = content.decode("utf-8", errors="ignore")
    if "html" in lowered or "<html" in decoded[:500].lower():
        return BeautifulSoup(decoded, "html.parser").get_text(" ", strip=True)
    return decoded.strip()


def chunk_text(text: str, chunk_size: int = 1400, overlap: int = 200) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        chunks.append(normalized[start : start + chunk_size])
        start += max(1, chunk_size - overlap)
    return chunks
