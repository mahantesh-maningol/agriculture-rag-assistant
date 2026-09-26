import os

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from backend.cache.llm_cache import configure_llm_cache
from backend.guardrails.output_guardrail import create_grounding_guardrail
from backend.guardrails.input_guardrail import create_input_guardrail
from backend.observability.tracer import Tracer
from backend.utils.extract_tool_context import extract_tool_context

from backend.agents.agent1 import create_rag_agent
from backend.tools.search_pdf import create_search_pdf_tool
from backend.tools.get_weather import get_weather
from pathlib import Path

load_dotenv()
BASE_DIR = Path(__file__).resolve().parents[2]
VECTORSTORE_PATH = BASE_DIR / "backend" / "vectorstore"

class RAGService:

    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not configured")

        # Configure LLM cache
        configure_llm_cache()

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
        ]

        # Guardrails
        self.input_guardrail = create_input_guardrail(self.llm)
        self.grounding_guardrail = create_grounding_guardrail(self.llm)

        # Agent
        self.agent = create_rag_agent(
            self.llm,
            self.tools,
        )

    def ask(self, user_question: str):

        # Create a new tracer for every request
        tracer = Tracer()

        # Input guardrail
        with tracer.span(
            "input_guardrail",
            {"question": user_question},
        ) as span:

            guardrail_result = self.input_guardrail(user_question)

            span["metadata"]["allowed"] = guardrail_result.allowed
            span["metadata"]["reason"] = guardrail_result.reason

        if not guardrail_result.allowed:
            return {
                "success": False,
                "answer": "I can only help with agriculture and weather-related questions.",
                "reason": guardrail_result.reason,
            }

        # Agent
        with tracer.span(
            "agent",
            {"question": user_question},
        ) as span:

            result = self.agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": user_question,
                        }
                    ],
                }
            )

        messages = result["messages"]

        final_message = messages[-1]

        # Extract context retrieved by tools
        context = extract_tool_context(messages)

        if context:

            with tracer.span(
                "grounding_guardrail",
                {
                    "question": user_question,
                    "context": context,
                },
            ) as span:

                grounding_result = self.grounding_guardrail(
                    user_question,
                    context,
                    final_message.content,
                )

                span["metadata"]["grounded"] = grounding_result.grounded
                span["metadata"]["reason"] = grounding_result.reason

            if grounding_result.grounded:
                return {
                    "success": True,
                    "answer": final_message.content,
                }

            return {
                "success": False,
                "answer": (
                    "I couldn't provide that answer because "
                    "the information could not be verified "
                    "against the agriculture knowledge base."
                ),
                "reason": grounding_result.reason,
            }

        # No search_pdf was used.
        # This can be a weather request.
        return {
            "success": True,
            "answer": final_message.content,
        }
