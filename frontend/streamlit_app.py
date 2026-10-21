import json
import os

import requests
import streamlit as st


# You can override this with BACKEND_URL in your environment.
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
CHAT_URL = f"{BACKEND_URL}/chat"
IMAGE_CHAT_URL = f"{BACKEND_URL}/chat/image"
ALLOWED_IMAGE_TYPES = ["jpg", "jpeg", "png", "webp"]
REQUEST_TIMEOUT_SECONDS = 120


st.set_page_config(
    page_title="Agriculture AI Assistant",
    page_icon="🌱",
    layout="centered",
)

st.title("🌱 Agriculture AI Assistant")
st.caption(
    "Ask crop-related questions or upload a leaf image for analysis and "
    "context-aware recommendations."
)

with st.form("agriculture_chat_form", clear_on_submit=False):
    user_question = st.text_input(
        "Your question",
        placeholder="e.g. What is wrong with this leaf, and how can I manage it?",
        help="You can ask a text-only question or include an image below.",
    )

    uploaded_image = st.file_uploader(
        "Upload a crop leaf image (optional)",
        type=ALLOWED_IMAGE_TYPES,
        accept_multiple_files=False,
        help="Supported formats: JPG, JPEG, PNG and WEBP.",
    )

    if uploaded_image is not None:
        st.image(
            uploaded_image,
            caption=f"Selected image: {uploaded_image.name}",
            use_container_width=True,
        )

    submitted = st.form_submit_button(
        "Analyze image" if uploaded_image is not None else "Send question",
        type="primary",
        use_container_width=True,
    )


def show_analysis(analysis):
    """Render prediction details without assuming a particular response schema."""
    if not analysis:
        return

    st.subheader("Leaf analysis")

    # PredictionResponse may be a dictionary, list, or another JSON-compatible
    # structure. Display structured output safely without assuming field names.
    if isinstance(analysis, dict):
        # Display common prediction fields prominently when present.
        preferred_fields = [
            "predicted_class",
            "class_name",
            "disease",
            "label",
            "confidence",
            "confidence_score",
        ]
        displayed = set()

        for field in preferred_fields:
            if field in analysis and analysis[field] is not None:
                value = analysis[field]
                st.write(f"**{field.replace('_', ' ').title()}:** {value}")
                displayed.add(field)

        remaining = {
            key: value
            for key, value in analysis.items()
            if key not in displayed
        }
        if remaining:
            with st.expander("Full prediction and product details"):
                st.json(remaining)
    else:
        with st.expander("Full prediction details", expanded=True):
            st.json(analysis)


if submitted:
    if uploaded_image is None and not user_question.strip():
        st.warning("Please enter a question or upload a leaf image.")
        st.stop()

    try:
        with st.spinner(
            "Analyzing the leaf and preparing recommendations..."
            if uploaded_image is not None
            else "Preparing your answer..."
        ):
            if uploaded_image is not None:
                # The controller expects multipart/form-data:
                # `question` as a form field and `file` as the uploaded image.
                uploaded_image.seek(0)
                files = {
                    "file": (
                        uploaded_image.name,
                        uploaded_image.getvalue(),
                        uploaded_image.type or "application/octet-stream",
                    )
                }
                data = {"question": user_question.strip()}

                response = requests.post(
                    IMAGE_CHAT_URL,
                    data=data,
                    files=files,
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
            else:
                # Preserve the existing text-only JSON request.
                response = requests.post(
                    CHAT_URL,
                    json={"question": user_question.strip()},
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )

        # Show useful backend validation/error messages where possible.
        if not response.ok:
            try:
                error_body = response.json()
                detail = error_body.get("detail", error_body)
            except (ValueError, AttributeError):
                detail = response.text or response.reason

            st.error(f"Backend returned HTTP {response.status_code}: {detail}")
            st.stop()

        try:
            result = response.json()
        except ValueError:
            st.error("The backend returned a response that was not valid JSON.")
            st.stop()

        if result.get("success"):
            answer = result.get("answer")
            if answer:
                st.subheader("Assistant response")
                st.markdown(answer)
            else:
                st.info("The analysis completed, but the assistant returned no text.")
        else:
            reason = result.get("reason")
            answer = result.get("answer")
            if answer:
                st.warning(answer)
            if reason:
                st.caption(f"Details: {reason}")
            if not answer and not reason:
                st.warning("The backend could not complete the request.")

        if uploaded_image is not None:
            show_analysis(result.get("analysis"))

    except requests.exceptions.Timeout:
        st.error(
            "The request timed out. Image analysis can take longer; "
            "please try again or increase REQUEST_TIMEOUT_SECONDS."
        )
    except requests.exceptions.ConnectionError:
        st.error(
            f"Unable to connect to the backend at {BACKEND_URL}. "
            "Make sure your FastAPI server is running."
        )
    except requests.exceptions.RequestException as error:
        st.error(f"Request failed: {error}")
    except Exception as error:
        st.error(f"Unexpected error while processing the request: {error}")
