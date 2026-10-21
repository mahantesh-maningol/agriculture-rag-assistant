# backend/schemas/predict_shema.py
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TopPrediction(BaseModel):
    label: str
    category: str
    confidence: float


class Recommendation(BaseModel):
    success: bool
    # verified_catalogue_matches | no_verified_products | needs_review |
    # no_treatment_indicated | catalogue_unavailable
    recommendation_status: Optional[str] = None
    prediction_class: Optional[str] = None
    condition: Optional[str] = None
    category: Optional[str] = None
    confidence: Optional[float] = None   # fraction 0-1, as returned by the tool
    message: Optional[str] = None
    products: List[Dict[str, Any]] = Field(default_factory=list)


class PredictionResponse(BaseModel):
    disease: str                          # class name, or "Unknown" when confidence is low
    category: str
    confidence: float                     # percent
    message: Optional[str] = None
    top_predictions: List[TopPrediction]
    recommendation: Recommendation