import os
import pdfplumber
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    import streamlit as st
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

KNOWLEDGE_DIR = "knowledge"
VECTORSTORE_DIR = "backend/vectorstore"


def load_pdfs_from_knowledge():
    documents = []

    for filename in os.listdir(KNOWLEDGE_DIR):

        if not filename.lower().endswith(".pdf"):
            continue

        file_path = os.path.join(KNOWLEDGE_DIR, filename)

        print(f"Reading: {filename}")

        with pdfplumber.open(file_path) as pdf:

            for page_number, page in enumerate(
                pdf.pages,
                start=1
            ):

                page_text = page.extract_text()

                if page_text:

                    documents.append(
                        Document(
                            page_content=page_text,
                            metadata={
                                "source": filename,
                                "page": page_number
                            }
                        )
                    )

    return documents


def create_vector_store():

    # ---------------------------------------------
    # 1. Load documents
    # ---------------------------------------------

    documents = load_pdfs_from_knowledge()

    print(f"\nTotal pages loaded: {len(documents)}")

    if not documents:
        raise ValueError(
            "No PDF files found in the knowledge directory."
        )

    # ---------------------------------------------
    # 2. Split documents into chunks
    # ---------------------------------------------

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Total chunks created: {len(chunks)}")

    # ---------------------------------------------
    # 3. Create embeddings
    # ---------------------------------------------

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY
    )

    # ---------------------------------------------
    # 4. Create FAISS vector store
    # ---------------------------------------------

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    # ---------------------------------------------
    # 5. Save FAISS vector store
    # ---------------------------------------------

    os.makedirs(
        VECTORSTORE_DIR,
        exist_ok=True
    )

    vector_store.save_local(
        VECTORSTORE_DIR
    )

    print(
        f"\nVector store saved to: {VECTORSTORE_DIR}"
    )


if __name__ == "__main__":
    create_vector_store()