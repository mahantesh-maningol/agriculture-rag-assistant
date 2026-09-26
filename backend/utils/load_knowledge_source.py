from pathlib import Path

import pdfplumber
from langchain_core.documents import Document


def load_pdfs_from_directory(directory: str) -> list[Document]:
    documents = []

    for pdf_path in Path(directory).glob("*.pdf"):
        print(f"Loading: {pdf_path.name}")

        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages):
                text = page.extract_text()

                if not text:
                    continue

                documents.append(Document( page_content=text, metadata={
                    "source": pdf_path.name,
                    "page": page_number,
                }))

    return documents