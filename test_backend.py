from backend.services.rag_service import RAGService

rag_service = RAGService()

question = "What are common pests affecting soybean?"
response = rag_service.ask(question)
print("\n--- RAG RESPONSE ---")
print(response)