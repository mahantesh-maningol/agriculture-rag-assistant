import logging
import os
import threading
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from backend.agents.agent1 import create_rag_agent
from backend.guardrails.input_guardrail import create_input_guardrail
from backend.tools.get_weather import get_weather
from backend.tools.leaf_analysis_tool import analyze_leaf_image
from backend.tools.product_recommendation_tool import (
    lookup_product_details,
    recommend_tomato_products,
)
from backend.tools.search_pdf import create_search_pdf_tool

logger = logging.getLogger(__name__)

load_dotenv()
BASE_DIR = Path(__file__).resolve().parents[2]
VECTORSTORE_PATH = BASE_DIR / "backend" / "vectorstore"
MAX_HISTORY_MESSAGES = 20  # 10 user/assistant turns; tool internals are not persisted.
MAX_ACTIVE_CONVERSATIONS = 500


class RAGService:
    """Existing tool-using RAG agent with short-lived conversational memory.

    Memory is in-process only. Use Redis/Postgres plus LangGraph checkpointers for
    durable, multi-worker production conversations.
    """

    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not configured")

        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=self.openai_api_key,
        )
        self.vector_store = FAISS.load_local(
            VECTORSTORE_PATH,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        self.retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4},
        )
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=1000,
            openai_api_key=self.openai_api_key,
        )

        self.search_pdf = create_search_pdf_tool(self.retriever, self.llm)
        self.tools = [
            self.search_pdf,
            get_weather,
            recommend_tomato_products,
            lookup_product_details,
            analyze_leaf_image,
        ]
        self.input_guardrail = create_input_guardrail(self.llm)
        self.agent = create_rag_agent(self.llm, self.tools)

        self._conversations: dict[str, list[dict[str, str]]] = {}
        self._history_lock = threading.RLock()

    @staticmethod
    def _history_context(history: list[dict[str, str]]) -> str:
        """Small context for classifying short follow-ups such as 'how much?'"""
        recent = history[-6:]
        return "\n".join(
            f"{'User' if item['role'] == 'user' else 'Assistant'}: "
            f"{item['content'][:700]}"
            for item in recent
        )

    @staticmethod
    def _log_tool_activity(messages: list) -> None:
        """Log which tools the agent called, with arguments and a result preview.

        History messages never contain tool calls, so scanning the whole list only
        reports the calls made during this request.
        """
        called = False
        for message in messages:
            for call in getattr(message, "tool_calls", None) or []:
                called = True
                logger.info("TOOL CALL   %s args=%s", call.get("name"), call.get("args"))
            if getattr(message, "type", None) == "tool":
                logger.info(
                    "TOOL RESULT %s: %s",
                    getattr(message, "name", "?"),
                    str(message.content)[:300],
                )
        if not called:
            logger.info("NO TOOL CALLED: the model answered from its own knowledge.")

    def reset_conversation(self, conversation_id: str) -> None:
        with self._history_lock:
            self._conversations.pop(conversation_id, None)

    def ask(
        self,
        user_question: str,
        context: str | None = None,
        conversation_id: str | None = None,
    ) -> dict[str, Any]:
        conversation_id = (conversation_id or uuid.uuid4().hex).strip()
        if not conversation_id or len(conversation_id) > 128:
            return {
                "success": False,
                "answer": "Invalid conversation identifier.",
                "reason": "invalid_conversation_id",
                "conversation_id": conversation_id,
            }

        with self._history_lock:
            history = list(self._conversations.get(conversation_id, []))

        # Run the injection/scope check on the actual current user text. Supply
        # recent dialogue separately so contextual follow-ups can be classified.
        guardrail_result = self.input_guardrail(
            user_question,
            context=self._history_context(history),
        )
        if not guardrail_result.allowed:
            return {
                "success": False,
                "answer": "I can help with agriculture, crop care, product use, and weather-related questions.",
                "reason": guardrail_result.reason,
                "conversation_id": conversation_id,
            }

        user_content = f"{user_question}\n\n{context}" if context else user_question
        messages = history + [{"role": "user", "content": user_content}]
        result = self.agent.invoke({"messages": messages})
        self._log_tool_activity(result["messages"])
        final_message = result["messages"][-1]
        answer = final_message.content
        if not isinstance(answer, str):
            answer = str(answer)

        # Persist only the user's turn and final answer, not intermediate tool
        # calls/results. The next invocation still has conversational context,
        # while avoiding incomplete tool-call sequences in message history.
        updated_history = history + [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": answer},
        ]
        updated_history = updated_history[-MAX_HISTORY_MESSAGES:]

        with self._history_lock:
            if conversation_id not in self._conversations and len(self._conversations) >= MAX_ACTIVE_CONVERSATIONS:
                oldest_id = next(iter(self._conversations))
                self._conversations.pop(oldest_id, None)
            self._conversations[conversation_id] = updated_history

        return {
            "success": True,
            "answer": answer,
            "conversation_id": conversation_id,
        }