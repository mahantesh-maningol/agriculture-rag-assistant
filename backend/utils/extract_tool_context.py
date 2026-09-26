from langchain_core.messages import ToolMessage


def extract_tool_context(messages):
    contexts = []
    for message in messages:
        if isinstance(message, ToolMessage) and message.name == "search_pdf":
            contexts.append(message.content)
    return "\n\n".join(contexts)
