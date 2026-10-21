import os

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from backend.guardrails.input_guardrail import create_input_guardrail

from backend.agents.agent1 import create_rag_agent
from backend.tools.search_pdf import create_search_pdf_tool
from backend.tools.get_weather import get_weather
from pathlib import Path
from backend.tools.product_recommendation_tool import (
    recommend_tomato_products,
)
from backend.tools.leaf_analysis_tool import analyze_leaf_image

load_dotenv()
BASE_DIR = Path(__file__).resolve().parents[2]
VECTORSTORE_PATH = BASE_DIR / "backend" / "vectorstore"

class RAGService:

    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not configured")


        # Embeddings
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=self.openai_api_key,
        )

        # Load existing FAISS vectorstore
        self.vector_store = FAISS.load_local(
            VECTORSTORE_PATH,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

        # Retriever
        self.retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4},
        )

        # LLM
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=1000,
            openai_api_key=self.openai_api_key,
        )

        # Tools
        self.search_pdf = create_search_pdf_tool(
            self.retriever,
            self.llm,
        )

        self.tools = [
            self.search_pdf,
            get_weather,
            recommend_tomato_products,
            analyze_leaf_image
        ]

        # Guardrails
        self.input_guardrail = create_input_guardrail(self.llm)

        # Agent
        self.agent = create_rag_agent(
            self.llm,
            self.tools,
        )

    def ask(self, user_question: str, context: str | None = None):
        # The guardrail only sees what the user actually typed.
        # `context` (e.g. a leaf image analysis) is added for the agent afterwards.
        guardrail_result = self.input_guardrail(user_question)
        if not guardrail_result.allowed:
            return {
                "success": False,
                "answer": "I can only help with agriculture and weather-related questions.",
                "reason": guardrail_result.reason,
            }

        result = self.agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"{user_question}\n\n{context}" if context else user_question
                        ),
                    }
                ],
            }
        )

        messages = result["messages"]

        final_message = messages[-1]
        return {
            "success": True,
            "answer": final_message.content,
        }