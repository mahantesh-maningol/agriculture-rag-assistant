from pydantic import BaseModel, Field


class GroundingResult(BaseModel):
    grounded: bool = Field(
        description="Whether the answer is supported by the provided context"
    )
    reason: str = Field(description="Explanation for the grounding decision")


def create_grounding_guardrail(llm):

    validator_llm = llm.with_structured_output(GroundingResult)

    def validate(question: str, context: str, answer: str) -> GroundingResult:

        prompt = f"""
You are a grounding validator for an agriculture RAG application.

Your job is to determine whether the assistant's answer
is supported by the retrieved knowledge-base context.

Rules:

1. Treat the retrieved context as the only source of truth.

2. Do not use your own general knowledge to validate the answer.

3. Every important factual claim in the answer must be
   supported by the retrieved context.

4. If the answer contains information that is not supported
   by the context, mark it as not grounded.

5. Do not judge whether the answer is generally true.
   Judge only whether it is supported by the supplied context.

User question:
{question}

Retrieved context:
{context}

Assistant answer:
{answer}
"""

        return validator_llm.invoke(prompt)

    return validate
