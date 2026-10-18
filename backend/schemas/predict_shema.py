from pydantic import BaseModel
from typing import List, Optional

"""Schema for prediction response."""

class PredictionResponse(BaseModel):
    disease: str
    confidence: float
    cause: Optional[str] = None
    treatment: Optional[List[str]] = None