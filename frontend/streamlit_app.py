import os
import uuid
from typing import Any

import requests
import streamlit as st

API_BASE_URL = os.getenv("RAG_API_URL", "http://localhost:8000").rstrip("/")
CHAT_URL = f"{API_BASE_URL}/chat"
IMAGE_CHAT_URL = f"{API_BASE_URL}/chat/image"
ALLOWED_IMAGE_TYPES = ["jpg", "jpeg", "png", "webp"]
REQUEST_TIMEOUT_SECONDS = 120

st.set_page_config(
    page_title="Agriculture AI Assistant",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Small amount of CSS to make the app feel closer to a chat product without
# depending on custom Streamlit components.
st.markdown(
    """
    <style>
      .block-container { max-width: 920px; padding-top: 1.4rem; padding-bottom: 7rem; }
      [data-testid="stChatMessage"] { padding: 0.35rem 0; }
      .app-subtitle { color: var(--text-color); opacity: 0.72; margin-top: -0.5rem; }
      [data-testid="stSidebar"] { border-right: 1px solid rgba(128,128,128,.18); }
    </style>
    """,
    unsafe_allow_html=True,
)


def new_conversation() -> None:
    st.session_state.conversation_id = uuid.uuid4().hex
    st.session_state.messages = []


def init_state() -> None:
    if "conversation_id" not in st.session_state:
        new_conversation()
    if "messages" not in st.session_state:
        st.session_state.messages = []


def show_analysis(analysis: dict[str, Any] | None) -> None:
    if not isinstance(analysis, dict):
        return

    disease = analysis.get("disease") or analysis.get("label") or "Not available"
    category = analysis.get("category") or "Not available"
    confidence = analysis.get("confidence")
    recommendation = analysis.get("recommendation") or {}
    products = recommendation.get("products") or []

    with st.expander("Leaf analysis details", expanded=False):
        col1, col2 = st.columns(2)
        col1.metric("Model prediction", str(disease))
        col2.metric(
            "Model confidence",
            f"{confidence}%" if confidence is not None else "Not available",
        )
        st.caption(f"Category: {category}")
        if recommendation.get("message"):
            st.info(recommendation["message"])
        if products:
            st.markdown("**Catalogue matches**")
            for product in products:
                st.markdown(f"**{product.get('name', 'Unnamed product')}**")
                if product.get("brand"):
                    st.caption(str(product["brand"]))
                if product.get("description"):
                    st.write(product["description"])
                details = []
                for field, label in (
                    ("application_rate", "Rate"),
                    ("rate_unit", "Rate unit"),
                    ("application_method", "Method"),
                    ("dosage_basis", "Dosage basis"),
                    ("water_volume_l_per_acre", "Water volume (L/acre)"),
                ):
                    value = product.get(field)
                    if value not in (None, ""):
                        details.append(f"**{label}:** {value}")
                if details:
                    st.markdown("  \n".join(details))
                st.divider()
        elif recommendation:
            st.caption("No verified product cards were returned for this analysis.")


def render_message(message: dict[str, Any]) -> None:
    role = message.get("role", "assistant")
    with st.chat_message("user" if role == "user" else "assistant"):
        if message.get("image_bytes"):
            st.image(message["image_bytes"], caption=message.get("image_name", "Uploaded leaf image"), width=320)
        if message.get("content"):
            st.markdown(message["content"])
        if role != "user":
            show_analysis(message.get("analysis"))


def call_text_chat(question: str) -> dict[str, Any]:
    response = requests.post(
        CHAT_URL,
        params={"conversation_id": st.session_state.conversation_id},
        json={"question": question},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def call_image_chat(question: str, uploaded_file: Any) -> dict[str, Any]:
    uploaded_file.seek(0)
    response = requests.post(
        IMAGE_CHAT_URL,
        data={
            "question": question,
            "conversation_id": st.session_state.conversation_id,
        },
        files={
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type or "application/octet-stream",
            )
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


init_state()

with st.sidebar:
    st.markdown("## 🌱 Agriculture AI")
    if st.button("＋ New chat", use_container_width=True, type="primary"):
        new_conversation()
        st.rerun()
    st.divider()
    st.markdown("**Things you can ask**")
    st.markdown(
        "- Upload a leaf photo for analysis\n"
        "- Ask about a detected condition\n"
        "- Ask about a product's label or application rate\n"
        "- Ask for a weather forecast or spray window\n"
        "- Ask questions from the agriculture knowledge base"
    )
    st.divider()
    st.caption("Conversation history is kept by this browser session and the running API process. It is not durable storage yet.")

st.title("🌱 Agriculture AI Assistant")
st.markdown(
    '<p class="app-subtitle">Ask farming questions, analyse a leaf, and continue with follow-up questions in the same conversation.</p>',
    unsafe_allow_html=True,
)

if not st.session_state.messages:
    st.markdown("### How can I help with your crop today?")
    st.markdown("You can start with a question or attach a leaf image in the message box below.")

for saved_message in st.session_state.messages:
    render_message(saved_message)

# Streamlit's native chat input supports attachments in recent Streamlit versions.
submitted = st.chat_input(
    "Message Agriculture AI…",
    accept_file="multiple",
    file_type=ALLOWED_IMAGE_TYPES,
)

if submitted:
    question = (submitted.text or "").strip()
    files = list(submitted.files or [])
    image_file = files[0] if files else None

    if len(files) > 1:
        st.warning("Please attach one leaf image at a time. The first image will be analysed.")

    if not question and image_file is None:
        st.warning("Type a question or attach a leaf image.")
        st.stop()

    if image_file is not None:
        image_bytes = image_file.getvalue()
        user_text = question or "Please analyse this leaf and explain what you find."
        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_text,
                "image_bytes": image_bytes,
                "image_name": image_file.name,
            }
        )
    else:
        user_text = question
        st.session_state.messages.append({"role": "user", "content": user_text})

    try:
        with st.spinner("Thinking…"):
            result = (
                call_image_chat(question, image_file)
                if image_file is not None
                else call_text_chat(question)
            )

        answer = result.get("answer") or "I couldn't generate an answer for this turn."
        assistant_message = {
            "role": "assistant",
            "content": answer,
            "analysis": result.get("analysis"),
        }
        if not result.get("success") and result.get("reason"):
            assistant_message["content"] += f"\n\n_Request status: {result['reason']}_"
        st.session_state.messages.append(assistant_message)
    except requests.exceptions.HTTPError as error:
        detail = "The API returned an error."
        try:
            detail = error.response.json().get("detail", detail)
        except Exception:
            pass
        st.session_state.messages.append(
            {"role": "assistant", "content": f"I couldn't process that request: {detail}"}
        )
    except requests.exceptions.RequestException as error:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": (
                    "I couldn't connect to the API. Check that FastAPI is running at "
                    f"`{API_BASE_URL}`. Details: `{error}`"
                ),
            }
        )

    st.rerun()
