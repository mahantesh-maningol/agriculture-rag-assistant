import os

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel
from typing import List, Optional
from backend.schemas.predict_shema import PredictionResponse
from backend.services.predict_service import predict_image

router = APIRouter(prefix="/predict", tags=["Predict"])

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

@router.post("", response_model=PredictionResponse)
def predict(file: UploadFile = File(..., description="Leaf image (jpg/png/webp)")):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Only jpg, png or webp images are allowed")

    try:
        return predict_image(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process image: {e}")