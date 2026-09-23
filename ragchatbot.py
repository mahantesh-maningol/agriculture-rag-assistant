import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from cache.llm_cache import configure_llm_cache
from guardrails.output_guardrail import create_grounding_guardrail
from observability.tracer import Tracer
from utils.extract_tool_context import extract_tool_context
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from agents.agent1 import create_rag_agent
from guardrails.input_guardrail import create_input_guardrail
from tools.search_pdf import create_search_pdf_tool
from tools.get_weather import get_weather

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

st.header("Agriculture AI Assistant")

configure_llm_cache()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY
)

vector_store = FAISS.load_local(
    "vectorstore", embeddings, allow_dangerous_deserialization=True
)

# get user question
user_question = st.text_input("Type your question here")

# generate answer
# question -> embeddings -> similarity search -> results to LLM -> response (CHAIN)

retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 4})


# define the LLM and Prompt instructions
llm = ChatOpenAI(
    model="gpt-4o-mini", temperature=0.3, max_tokens=1000, openai_api_key=OPENAI_API_KEY
)

search_pdf = create_search_pdf_tool(retriever, llm)
tools = [search_pdf, get_weather]

input_guardrail = create_input_guardrail(llm)
grounding_guardrail = create_grounding_guardrail(llm)

agent = create_rag_agent(llm, tools)
if user_question:
    # Create a NEW tracer for this request
    tracer = Tracer()
    with tracer.span("input_guardrail", {"question": user_question}) as span:
        guardrail_result = input_guardrail(user_question)
        span["metadata"]["allowed"] = guardrail_result.allowed
        span["metadata"]["reason"] = guardrail_result.reason

    if not guardrail_result.allowed:
        st.warning("I can only help with agriculture and weather-related questions.")
    else:
        # docs = retriever.invoke(user_question)

        # st.write("Retrieved Chunks:")
        with tracer.span("agent", {"question": user_question}) as span:
            result = agent.invoke(
                {
                    "messages": [{"role": "user", "content": user_question}],
                }
            )

        messages = result["messages"]

        # st.write("messages:")
        final_message = messages[-1]

        context = extract_tool_context(messages)
        if context:
            with tracer.span(
                "grounding_guardrail", {"question": user_question, "context": context}
            ) as span:
                grounding_result = grounding_guardrail(
                    user_question, context, final_message.content
                )
                span["metadata"]["grounded"] = grounding_result.grounded
                span["metadata"]["reason"] = grounding_result.reason

            if grounding_result.grounded:
                st.write(final_message.content)
            else:
                st.warning(
                    "I couldn't provide that answer because "
                    "the information could not be verified "
                    "against the agriculture knowledge base."
                )
        else:
            # No search_pdf was used.
            # This could be a weather request.
            st.write(final_message.content)

    # trace = tracer.get_trace()

    # print("\n===== AI TRACE =====")

    # print(trace)


# structured_llm = llm.with_structured_output(RAGResponse)

# llm_with_tools = llm.bind_tools(tools)

# response = llm_with_tools.invoke(user_question)

# st.write("Response:", response)
#
# tool_call = response.tool_calls[0]
#
# tool_result = search_pdf.invoke(tool_call["args"])

# st.write("tool_result:", tool_result)

# provide the prompts
# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "You are a financial analyst assistant.\n\n"
#             "You have access to the following tools:\n"
#             "- search_pdf: Search the provided Amazon quarterly results document.\n"
#             "- get_weather: Get current weather information for a location.\n\n"
#             "Follow these rules strictly:\n"
#             "1. For questions about the Amazon quarterly results document, "
#             "use the search_pdf tool to retrieve the relevant information.\n"
#             "2. For weather-related questions, use the get_weather tool.\n"
#             "3. Do not use your general knowledge when answering questions "
#             "about the Amazon quarterly results document.\n"
#             "4. Do not invent or estimate financial figures.\n"
#             "5. When mentioning financial numbers, include the relevant period.\n"
#             "6. If search_pdf does not return information relevant to the question, "
#             "say: I couldn't find this information in the provided document.\n"
#             "7. Keep the answer clear and concise.\n"
#             "8. Confidence must be between 0 and 1.",
#             # "Context:\n{context}"
#         ),
#         ("human", "{question}"),
#     ]
# )

# chain_for_tools = prompt | llm_with_tools #| StrOutputParser()


# chain = (
#     {"context":retriever | format_docs, "question": RunnablePassthrough()} | prompt | structured_llm  ##llm ## | StrOutputParser()
# )
# for i, doc in enumerate(docs):
#     st.write(f"----chunk {i + 1}----")
#     st.write(doc.page_content)

# response = chain.invoke(user_question)
# st.write("Response:")
# st.write(response)

# messages = prompt.invoke({
#    "question": user_question
# }).to_messages()

# response = structured_llm.invoke(prompt_for_tools)
# response = chain_for_tools.invoke({
#    # "context": tool_result,
#    "question": user_question,
# })
# Inspect agent messages
# for message in result["messages"]:
#     st.write(message)


## Agent will handle this loop
# if response.tool_calls:
#     messages.append(response)
#     for tool_call in response.tool_calls:
#         if tool_call["name"] == "search_pdf":
#             tool_result = search_pdf.invoke(tool_call["args"])
#         elif tool_call["name"] == "get_weather":
#             tool_result = get_weather.invoke(tool_call["args"])
#         else:
#             continue
#         # messages.append(response)
#         messages.append(
#             ToolMessage(
#                 content=str(tool_result),
#                 tool_call_id=tool_call["id"]
#             )
#         )
#     final_response = llm_with_tools.invoke(messages)
# else:
#     final_response = response

# st.write("Response:")
# st.write(final_response.content)

# st.write("Response:")
# st.write(response)

# st.write("### Answer")
# st.write(response.answer)
#
# st.write("### Confidence")
# st.write(response.confidence)
#
# st.write("### Sources")
# for source in response.sources:
#     st.write(source)


# st.write("Token Usage:")
# st.write(response.usage_metadata)
