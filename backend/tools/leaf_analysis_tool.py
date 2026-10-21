
import json
import os
from pathlib import Path

from langchain_core.tools import tool

from backend.services.leaf_analysis_service import (
    analyze_leaf,
    format_for_agent,
)

UPLOAD_DIR = Path(
    os.getenv("LEAF_UPLOAD_DIR", "/tmp/agriculture_leaf_uploads")
).resolve()


@tool("analyze_leaf_image")
def analyze_leaf_image(image_path: str) -> str:
    """Analyze an uploaded crop leaf image.

    Returns the predicted condition, confidence, top predictions,
    and verified product recommendations.

    image_path must be a server-generated path inside the configured
    leaf upload directory.
    """
    try:
        image = Path(image_path).resolve()
        image.relative_to(UPLOAD_DIR)
    except (TypeError, ValueError, OSError):
        return json.dumps({
            "success": False,
            "error": "Invalid uploaded image path."
        })

    if not image.is_file():
        return json.dumps({
            "success": False,
            "error": "Uploaded image was not found."
        })

    try:
        analysis = analyze_leaf(str(image))
        return format_for_agent(analysis)
    except ValueError as exc:
        return json.dumps({
            "success": False,
            "error": str(exc)
        })
    except Exception:
        return json.dumps({
            "success": False,
            "error": "Leaf image analysis failed."
        })
