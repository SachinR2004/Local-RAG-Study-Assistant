
from pypdf import PdfReader


def extract_pdf_pages(pdf_path: str) -> list[dict]:
    """
    Extract text from a PDF while preserving page numbers.

    Returns:
        A list of dictionaries containing:
        - page: Original PDF page number (1-indexed)
        - text: Extracted page text
    """

    reader = PdfReader(pdf_path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            pages.append({
                "page": page_number,
                "text": text.strip()
            })

    return pages


def extract_text_from_pdf(pdf_path: str) -> str:

    pages = extract_pdf_pages(pdf_path)

    return "\n".join(page["text"] for page in pages)