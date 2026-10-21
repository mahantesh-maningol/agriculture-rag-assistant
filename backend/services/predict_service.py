import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError

BASE_DIR = Path(__file__).resolve().parent          # backend/services
MODEL_PATH = BASE_DIR.parent / "saved_model" / "model.keras"
CLASSES_PATH = BASE_DIR.parent / "saved_model" / "classes.json"

IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 70.0   # keep equal to MINIMUM_CONFIDENCE * 100 in product_recommendation_tool.py
TOP_K = 3

# Class names that are pests rather than diseases (everything with "deficiency"
# in its name is a nutrient problem, "Healthy" is healthy, the rest are diseases)
PEST_CLASSES = {"Spider_mites", "Mites", "Leaf_miner", "Jassids_and_mites"}

# ---- Load static assets once at import ----
with open(CLASSES_PATH, "r") as f:
    class_indices = json.load(f)                    # {"ClassName": index}

# Order by index so CLASS_NAMES[i] always matches the model's output i
CLASS_NAMES = [name for name, _ in sorted(class_indices.items(), key=lambda kv: kv[1])]

model = tf.keras.models.load_model(MODEL_PATH)

# Take the input size from the model itself, so it always matches train.py
IMG_SIZE = tuple(model.input_shape[1:3])

# Fail loudly at startup if model and classes.json come from different training runs
assert model.output_shape[-1] == len(CLASS_NAMES), (
    f"Model has {model.output_shape[-1]} outputs but classes.json has {len(CLASS_NAMES)} classes"
)


def _category(class_name: str) -> str:
    if class_name == "Healthy":
        return "Healthy"
    if "deficiency" in class_name.lower():
        return "Nutrient deficiency"
    if class_name in PEST_CLASSES:
        return "Pest damage"
    return "Disease"


def _preprocess(source) -> np.ndarray:
    """source: file path or file-like object.

    The model contains its own Rescaling layer, so we pass RAW 0-255 pixels here.
    Do NOT divide by 255, that would be applied twice.
    """
    try:
        img = Image.open(source)
        img = ImageOps.exif_transpose(img).convert("RGB")   # fixes rotated phone photos
    except (UnidentifiedImageError, OSError) as e:
        raise ValueError("Uploaded file is not a valid image") from e

    arr = np.asarray(img, dtype=np.float32)
    arr = tf.image.resize(arr, IMG_SIZE).numpy()             # same resize op used in training
    return np.expand_dims(arr, axis=0)


def predict_image(source) -> dict:
    batch = _preprocess(source)

    preds = model(batch, training=False).numpy()[0]
    idx = int(np.argmax(preds))
    confidence = round(float(preds[idx]) * 100, 2)

    top_predictions = [
        {
            "label": CLASS_NAMES[i],
            "category": _category(CLASS_NAMES[i]),
            "confidence": round(float(preds[i]) * 100, 2),
        }
        for i in np.argsort(preds)[::-1][:TOP_K]
    ]

    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "disease": "Unknown",
            "category": "Unknown",
            "confidence": confidence,
            "message": "Low confidence. Please upload a clear, close-up photo of a single leaf.",
            "top_predictions": top_predictions,
        }

    predicted_class = CLASS_NAMES[idx]

    return {
        "disease": predicted_class,            # key kept so existing API consumers still work
        "category": _category(predicted_class),
        "confidence": confidence,
        "top_predictions": top_predictions,
    }


if __name__ == "__main__":
    print("Prediction:", predict_image("test.jpg"))