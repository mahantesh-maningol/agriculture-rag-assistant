import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent          # backend/services
MODEL_PATH = BASE_DIR.parent / "saved_model" / "model.h5"
CLASSES_PATH = BASE_DIR.parent / "saved_model" / "classes.json"
TREATMENT_PATH = BASE_DIR / "treatment.json"

IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 70.0

DEFAULT_TREATMENT = {"cause": "Unknown", "treatment": ["Consult agricultural expert"]}

# ---- Load static assets once at import ----
with open(TREATMENT_PATH, "r") as f:
    TREATMENT_MAP = json.load(f)

with open(CLASSES_PATH, "r") as f:
    class_indices = json.load(f)                    # {"ClassName": index}

# Order by index so CLASS_NAMES[i] always matches the model's output i
CLASS_NAMES = [name for name, _ in sorted(class_indices.items(), key=lambda kv: kv[1])]

model = tf.keras.models.load_model(MODEL_PATH)


def _preprocess(source) -> np.ndarray:
    """source: file path or file-like object."""
    img = Image.open(source).convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32) / 255.0   # must match training preprocessing
    return np.expand_dims(arr, axis=0)


def predict_image(source) -> dict:
    batch = _preprocess(source)

    preds = model(batch, training=False).numpy()[0]
    idx = int(np.argmax(preds))
    confidence = round(float(preds[idx]) * 100, 2)

    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "disease": "Unknown",
            "confidence": confidence,
            "cause": None,
            "treatment": None,
        }

    predicted_class = CLASS_NAMES[idx]
    info = TREATMENT_MAP.get(predicted_class, DEFAULT_TREATMENT)

    return {
        "disease": predicted_class,
        "confidence": confidence,
        "cause": info["cause"],
        "treatment": info["treatment"],
    }


if __name__ == "__main__":
    print("Prediction:", predict_image("test.jpg"))