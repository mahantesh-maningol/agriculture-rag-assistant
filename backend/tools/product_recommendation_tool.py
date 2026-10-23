# backend/tools/product_recommendation_tool.py

import json
import logging
import re

from langchain.tools import tool

from backend.database import get_connection
from backend.services.tomato_condition_mapping import TOMATO_CLASS_MAPPING

logger = logging.getLogger(__name__)

# Calibrate this threshold using your model's validation results.
# Keep it equal to CONFIDENCE_THRESHOLD / 100 in predict_service.py.
MINIMUM_CONFIDENCE = 0.70

PRODUCT_QUERY = """
    SELECT
        p.id,
        p.name,
        p.brand,
        p.product_type,
        p.description,
        p.nitrogen_pct,
        p.phosphorus_pct,
        p.potassium_pct,
        p.magnesium_pct,
        p.calcium_pct,
        p.sulfur_pct,
        p.micronutrients,
        p.target_conditions,
        p.application_rate,
        p.rate_unit,
        p.application_method,
        p.dosage_basis,
        p.water_volume_l_per_acre,
        p.product_image_url,
        p.product_guide_url,
        p.guide_document_id,
        p.label_source,
        p.label_verified
    FROM products AS p
    WHERE p.is_active = TRUE
      AND p.label_verified = TRUE
      AND EXISTS (
          SELECT 1
          FROM jsonb_array_elements(
              CASE
                  WHEN jsonb_typeof(p.target_conditions) = 'array'
                  THEN p.target_conditions
                  ELSE '[]'::jsonb
              END
          ) AS condition_item
          WHERE lower(trim(condition_item ->> 'name'))
                = lower(trim(%s))
            AND lower(trim(condition_item ->> 'category'))
                = lower(trim(%s))
      )
    ORDER BY p.name
    LIMIT 2;
"""


def get_tomato_product_recommendations(predicted_class: str, confidence: float) -> dict:
    """Plain function (no LLM). Used directly by the predict endpoint and,
    through the tool wrapper below, by the chat agent.

    confidence: a fraction (0.93) or a percentage (93).
    """

    # 1. Validate the model output.
    mapping = TOMATO_CLASS_MAPPING.get(predicted_class)

    if mapping is None:
        return {
            "success": False,
            "prediction_class": predicted_class,
            "message": (
                "Unsupported tomato prediction class. "
                "Do not invent a condition or recommend products."
            ),
            "products": [],
        }

    try:
        score = float(confidence)
    except (TypeError, ValueError):
        return {
            "success": False,
            "message": "Confidence must be a numeric value.",
            "products": [],
        }

    # Support either 0.93 or 93 for 93% confidence.
    if 1 < score <= 100:
        score = score / 100

    if not 0 <= score <= 1:
        return {
            "success": False,
            "message": "Confidence must be between 0 and 1, or 0 and 100.",
            "products": [],
        }

    condition = mapping["condition"]
    category = mapping["category"]

    # 2. Healthy leaves should never trigger treatment lookup.
    if category == "healthy":
        return {
            "success": True,
            "prediction_class": predicted_class,
            "condition": condition,
            "category": category,
            "confidence": round(score, 4),
            "recommendation_status": "no_treatment_indicated",
            "message": (
                "The model classified the leaf as healthy. "
                "No disease or deficiency product recommendation "
                "is indicated by this prediction."
            ),
            "products": [],
        }

    # 3. Low confidence requires caution, not automatic treatment.
    if score < MINIMUM_CONFIDENCE:
        return {
            "success": True,
            "prediction_class": predicted_class,
            "condition": condition,
            "category": category,
            "confidence": round(score, 4),
            "recommendation_status": "needs_review",
            "message": (
                "Prediction confidence is below the configured threshold. "
                "Request a clearer image or further inspection before "
                "making a product recommendation."
            ),
            "products": [],
        }

    # 4. Match the canonical condition against PostgreSQL.
    # Only label-verified records can be returned as recommendations.
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(PRODUCT_QUERY, (condition, category))
                rows = cur.fetchall()

        logger.info(
            "Product lookup: condition=%r category=%r -> %d row(s)",
            condition, category, len(rows),
        )

        products = [
            dict(row) if not isinstance(row, dict) else row
            for row in rows
        ]

        if not products:
            return {
                "success": True,
                "prediction_class": predicted_class,
                "condition": condition,
                "category": category,
                "confidence": round(score, 4),
                "recommendation_status": "no_verified_products",
                "message": (
                    "No matching label-verified products were found. "
                    "Do not invent a product, dosage, or treatment."
                ),
                "products": [],
            }

        return {
            "success": True,
            "prediction_class": predicted_class,
            "condition": condition,
            "category": category,
            "confidence": round(score, 4),
            "recommendation_status": "verified_catalogue_matches",
            "message": (
                "These products have verified labels and match the "
                "catalogue condition. Confirm tomato crop suitability, "
                "approved use, label directions, and safety restrictions "
                "before presenting an actionable recommendation."
            ),
            "products": products,
        }

    except Exception:
        logger.exception("Tomato product recommendation lookup failed")

        return {
            "success": False,
            "prediction_class": predicted_class,
            "condition": condition,
            "category": category,
            "recommendation_status": "catalogue_unavailable",
            "message": (
                "The product catalogue is temporarily unavailable. "
                "Do not fabricate product results."
            ),
            "products": [],
        }


@tool
def recommend_tomato_products(
    predicted_class: str,
    confidence: float,
) -> str:
    """
    Find label-verified products matching a tomato leaf image prediction.

    Only call this with values taken from a leaf-image prediction result.
    Never guess a class from the user's description and never estimate
    the confidence yourself.

    Args:
        predicted_class: Exact class name returned by the prediction API,
            such as Early_blight, Nitrogen_deficiency, or Healthy.
        confidence: Model confidence as a fraction from 0 to 1,
            or a percentage from 0 to 100.

    Returns:
        JSON with the mapped condition, confidence, and matching
        label-verified products. Healthy or unsupported predictions
        do not trigger product recommendations.
    """
    return json.dumps(
        get_tomato_product_recommendations(predicted_class, confidence),
        default=str,
    )


# --------------------------------------------------------------------------
# Look up a product BY NAME (rate, unit, method, composition, ...)
# --------------------------------------------------------------------------

LOOKUP_COLUMNS = """
    p.id, p.name, p.brand, p.product_type, p.description,
    p.nitrogen_pct, p.phosphorus_pct, p.potassium_pct, p.magnesium_pct,
    p.calcium_pct, p.sulfur_pct, p.micronutrients, p.target_conditions,
    p.application_rate, p.rate_unit, p.application_method, p.dosage_basis,
    p.water_volume_l_per_acre, p.product_image_url, p.product_guide_url
"""


def _like_pattern(token: str) -> str:
    """Escape LIKE wildcards so '50%' means a literal 50%, then wrap for 'contains'."""
    escaped = token.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def get_product_details(product_query: str, limit: int = 5) -> dict:
    """Plain function (no LLM): find verified products whose name, brand or type
    contains every word of `product_query`."""

    tokens = [
        t for t in (product_query or "").split()
        if re.search(r"[A-Za-z0-9]", t)          # drop stray '-' or ','
    ][:8]

    if not tokens:
        return {
            "success": False,
            "status": "invalid_query",
            "message": "Provide a product name, or part of one.",
            "products": [],
        }

    # Only fixed SQL text is built here; every user-supplied value goes through %s.
    word_clauses = " AND ".join(
        "(p.name ILIKE %s OR p.brand ILIKE %s OR p.product_type ILIKE %s)"
        for _ in tokens
    )
    params = []
    for token in tokens:
        pattern = _like_pattern(token)
        params += [pattern, pattern, pattern]

    sql = f"""
        SELECT {LOOKUP_COLUMNS}
        FROM products AS p
        WHERE p.is_active = TRUE
          AND p.label_verified = TRUE
          AND {word_clauses}
        ORDER BY (lower(p.name) = lower(%s)) DESC, p.name, p.brand
        LIMIT %s;
    """
    params += [product_query.strip(), limit]

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        products = [dict(r) if not isinstance(r, dict) else r for r in rows]
        logger.info("Product lookup by name: query=%r -> %d row(s)", product_query, len(products))

        if not products:
            return {
                "success": True,
                "status": "not_found",
                "query": product_query,
                "message": (
                    "No matching product in the verified catalogue. Say so plainly. "
                    "Do not describe the product from general knowledge."
                ),
                "products": [],
            }

        return {
            "success": True,
            "status": "found",
            "query": product_query,
            "message": (
                "Answer using ONLY these stored fields, exactly as written "
                "(application_rate with rate_unit, dosage_basis, application_method, "
                "water_volume_l_per_acre, composition, target_conditions). "
                "If a field is null, say it is not recorded. "
                "Do not add doses or facts from general knowledge. "
                "Remind the user to follow the product label."
            ),
            "products": products,
        }

    except Exception:
        logger.exception("Product lookup by name failed")
        return {
            "success": False,
            "status": "catalogue_unavailable",
            "message": "The product catalogue is temporarily unavailable. Do not guess product data.",
            "products": [],
        }


@tool
def lookup_product_details(product_query: str) -> str:
    """
    Look up a product in the verified product catalogue by name, brand or type.

    ALWAYS use this when the user names a product, or asks about a product's
    application rate, rate unit, dosage, application method, water volume,
    composition, target diseases/pests/deficiencies, or description.
    Answer only from the returned fields. Do not use the PDF knowledge base or
    general knowledge for product facts, and do not make up values that are missing.

    For weather questions about applying a product, call this first and use its
    application_method: 'Foliar spray' means application_type foliar for get_weather;
    'Soil application', 'Fertigation' or 'Soil drench' means application_type soil.
    Then you do not need to ask the user how the product is applied.

    Args:
        product_query: the product name or part of it, for example
            'Acibenzolar-S-methyl 50% WG - 1 kg' or 'potassium nitrate'.

    Returns:
        JSON with status found / not_found / catalogue_unavailable and the
        matching products (up to 5) with their stored fields.
    """
    logger.info("lookup_product_details called with %r", product_query)
    return json.dumps(get_product_details(product_query), default=str)