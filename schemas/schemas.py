from pydantic import BaseModel, Field

class Source(BaseModel):
    page: int = Field(description="Page number")
    reason: str = Field(description="Relevant source references")


class RAGResponse(BaseModel):
    answer: str = Field(description="Answer based only on the provided PDF context")
    confidence: float = Field(description="Confidence between 0 and 1")
    sources: list[Source]