from stock_agent.models import AgentResult
from stock_agent.services.pdf_service import extract_pdf_text


def answer_from_documents(question: str, pdf_bytes: bytes | None) -> AgentResult:
    if not pdf_bytes:
        return AgentResult("### Document analysis\nNo PDF was supplied.")
    text = extract_pdf_text(pdf_bytes)
    return AgentResult(
        f"### Document analysis\nPDF text extracted ({len(text):,} characters). "
        "Connect this step to the vector store and Bedrock prompt.",
        sources=["Uploaded PDF"],
    )
