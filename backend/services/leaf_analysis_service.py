# backend/services/leaf_analysis_service.py
"""
One entry point for "the user sent a leaf photo": predict, then look up products.

Used by the predict endpoint, and by the chat endpoint when an image is attached.
No LLM is involved here, so the result is deterministic and fast.
"""
import json
import logging

from backend.services.predict_service import CLASS_NAMES, predict_image
from backend.services.tomato_condition_mapping import TOMATO_CLASS_MAPPING
from backend.tools.product_recommendation_tool import get_tomato_product_recommendations

logger = logging.getLogger(__name__)

# Products are only suggested when the top class beats the runner-up by at least
# this many percentage points. The model's common confusions (Healthy vs
# Potassium_deficiency vs Mites) tend to show up as near-ties.
MIN_MARGIN = 15.0

# Fail at startup, not at the first user upload, if the model and mapping disagree.
_unmapped = [c for c in CLASS_NAMES if c not in TOMATO_CLASS_MAPPING]
if _unmapped:
    raise RuntimeError(f"Classes missing from TOMATO_CLASS_MAPPING: {_unmapped}")


def analyze_leaf(source) -> dict:
    """source: file path or file-like object (e.g. UploadFile.file).

    Returns the usual prediction fields (disease, category, confidence, cause,
    treatment, top_predictions) plus a "recommendation" key, so existing API
    consumers keep working. Raises ValueError if the file is not a valid image.
    """
    prediction = predict_image(source)

    top = prediction["top_predictions"]
    best = top[0]

    # Use the top class even when the prediction came back "Unknown", so the
    # recommender applies its own low-confidence handling (needs_review).
    recommendation = get_tomato_product_recommendations(
        best["label"], best["confidence"] / 100
    )

    if (
        recommendation.get("recommendation_status") == "verified_catalogue_matches"
        and len(top) > 1
        and best["confidence"] - top[1]["confidence"] < MIN_MARGIN
    ):
        recommendation = {
            **recommendation,
            "recommendation_status": "needs_review",
            "message": (
                f"The model is torn between {best['label']} and {top[1]['label']}. "
                "Ask for a clearer, close-up photo before recommending a product."
            ),
            "products": [],
        }

    return {**prediction, "recommendation": recommendation}


# Only these product fields are shown to the LLM. The full records (image and
# guide URLs, nutrient percentages, ...) still go back to the client untouched.
_AGENT_PRODUCT_FIELDS = (
    "name", "brand", "product_type", "description", "application_rate",
    "rate_unit", "application_method", "dosage_basis", "water_volume_l_per_acre",
)


def format_for_agent(analysis: dict) -> str:
    """Turn analyze_leaf() output into a context block for the chat agent."""
    rec = analysis["recommendation"]
    payload = {
        "detected": analysis["disease"],
        "category": analysis["category"],
        "confidence_percent": analysis["confidence"],
        "top_predictions": analysis["top_predictions"],
        "recommendation_status": rec.get("recommendation_status"),
        "recommendation_message": rec.get("message"),
        "products": [
            {k: p.get(k) for k in _AGENT_PRODUCT_FIELDS}
            for p in rec.get("products", [])
        ],
    }
    return (
        "[LEAF IMAGE ANALYSIS. The image model has already run; do not call "
        "recommend_tomato_products again.]\n"
        f"{json.dumps(payload, default=str, indent=2)}\n"
        "[INSTRUCTIONS: Answer using only the result above. Say what was detected "
        "and how confident the model is. If recommendation_status is needs_review "
        "(or detected is Unknown), say the photo was not clear enough and ask for a "
        "close-up of a single leaf, and do not suggest products. If it is "
        "no_treatment_indicated, say the leaf looks healthy. If it is "
        "verified_catalogue_matches, list the products using only the fields given; "
        "never invent a dose, rate or safety claim, and tell the user to read the "
        "product label and confirm it is approved for tomato before use. If it is "
        "no_verified_products or catalogue_unavailable, say so and do not name products.]"
    )