import streamlit as st
import requests

st.header("Agriculture AI Assistant")

user_question = st.text_input("Type your question here")

if user_question:
    try:

        response = requests.get(
            "http://localhost:8000/chat",
            json={
                "question": user_question
            }
        )

        result = response.json()

        if result["success"]:
            st.write(result["answer"])
        else:
            st.warning(result["answer"])

    except requests.exceptions.RequestException as error:
        st.error(f"Unable to connect to the backend: {error}")

    