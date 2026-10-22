# Conversational Agriculture RAG update

This update adds a ChatGPT-style Streamlit interface and short-term conversation memory to the existing FastAPI + LangChain RAG agent. It reuses the existing FAISS index, PDF search tool, weather tool, tomato product recommender, and leaf analysis service.


## Run locally

Terminal 1, from the project root:

```bash
uvicorn backend.app:app --reload --port 8000
```

Terminal 2, from the project root and in the same virtual environment:

```bash
streamlit run streamlit_app.py
```