
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "saved_model"
    / "best.pt"
)

leaf_model = YOLO(str(MODEL_PATH))


def validate_leaf_image(source) -> dict:
    try:
        if hasattr(source, "seek"):
            source.seek(0)

        if hasattr(source, "read"):
            image = Image.open(source).convert("RGB")
        else:
            image = Image.open(source).convert("RGB")

    except (UnidentifiedImageError, OSError, ValueError):
        return {
            "valid_leaf": False,
            "status": "rejected",
            "message": "Invalid image. Please upload a valid leaf photo.",
            "detections": [],
        }

    results = leaf_model.predict(
        source=image,
        conf=0.15,
        verbose=False,
    )

    result = results[0]
    boxes = result.boxes

    detections = []

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())

            detections.append({
                "class": result.names[class_id],
                "confidence": round(confidence, 4),
            })

    # Temporary diagnostic logging
    print("YOLO leaf validation detections:", detections)

    if not detections:
        return {
            "valid_leaf": False,
            "status": "rejected",
            "message": (
                "I couldn't confidently identify a leaf in this image. "
                "Please upload a clear, close-up photo of the leaf."
            ),
            "detections": [],
        }

    return {
        "valid_leaf": True,
        "status": "accepted",
        "message": "Leaf detected successfully.",
        "detections": detections,
    }
