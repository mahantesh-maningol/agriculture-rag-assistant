import logging

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.schemas.predict_shema import PredictionResponse
from backend.services.leaf_analysis_service import analyze_leaf

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["Predict"])

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


# Plain `def` on purpose: TensorFlow and the database call are blocking, so
# FastAPI runs this in a worker thread instead of freezing the event loop.
@router.post("", response_model=PredictionResponse)
def predict(file: UploadFile = File(..., description="Leaf image (jpg/png/webp)")):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Only jpg, png or webp images are allowed")

    try:
        return analyze_leaf(file.file)
    except ValueError as e:                       # not a valid image
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Leaf analysis failed")
        raise HTTPException(status_code=500, detail="Could not analyse the image")