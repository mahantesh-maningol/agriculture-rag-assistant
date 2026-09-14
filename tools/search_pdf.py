from langchain_core.tools import tool


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def create_search_pdf_tool(retriever):
    @tool
    def search_pdf(question: str) -> str:
        """Search the agriculture knowledge base and return relevant information."""
        docs = retriever.invoke(question)
        # return format_docs(docs)
        return "\n\n".join(
            f"[Source: {doc.metadata.get('source')}, "
            f"Page: {doc.metadata.get('page')}]\n"
            f"{doc.page_content}"
            for doc in docs
        )

    return search_pdf
