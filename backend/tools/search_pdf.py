import streamlit as st
from langchain_core.tools import tool

from backend.guardrails.indirect_injuction_guardrail import detect_indirect_injection



def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def create_search_pdf_tool(retriever, llm):
    @tool
    def search_pdf(question: str) -> str:
        """Search the agriculture knowledge base and return relevant information."""
        docs = retriever.invoke(question)
        isNotSafe = detect_indirect_injection(llm, docs)
        if isNotSafe:
            return (
                "The retrieved knowledge contains potentially unsafe "
                "instructions and cannot be used."
            )

        # return format_docs(docs)
        return "\n\n".join(
            f"[Source: {doc.metadata.get('source')}, "
            f"Page: {doc.metadata.get('page')}]\n"
            f"{doc.page_content}"
            for doc in docs
        )

    return search_pdf
