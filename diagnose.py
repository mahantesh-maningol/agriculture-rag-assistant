"""Evaluate the saved tomato model using dataset/{train,val,test}/<class>/.

Without source labels or filename prefixes, this script reports performance by
split and class, not separately by PlantVillage versus Tomato Village.
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
DATA = BASE_DIR / "dataset"
MODEL_PATH = BASE_DIR / "backend" / "saved_model" / "model.keras"
CLASSES_PATH = BASE_DIR / "backend" / "saved_model" / "classes.json"
THRESHOLD = 70.0
BATCH_SIZE = 32
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

model = tf.keras.models.load_model(MODEL_PATH)
IMG_SIZE = tuple(model.input_shape[1:3])
with open(CLASSES_PATH, encoding="utf-8") as f:
    class_indices = json.load(f)
class_names = [name for name, _ in sorted(class_indices.items(), key=lambda item: item[1])]


def collect_files(split):
    split_dir = DATA / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")
    files = []
    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        if class_dir.name not in class_indices:
            print(f"WARNING: ignoring unknown class folder {class_dir.name!r} in {split}")
            continue
        files.extend(p for p in class_dir.rglob("*")
                     if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    return files


def load_image(path):
    image = tf.io.decode_image(tf.io.read_file(str(path)), channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    return tf.image.resize(image, IMG_SIZE).numpy()


def predict(paths):
    outputs = []
    for start in range(0, len(paths), BATCH_SIZE):
        batch = np.stack([load_image(p) for p in paths[start:start + BATCH_SIZE]])
        outputs.append(model(batch, training=False).numpy())
    return np.concatenate(outputs) if outputs else np.empty((0, len(class_names)))


for split in ("train", "val", "test"):
    files = collect_files(split)
    if not files:
        print(f"\n{split}: no images found")
        continue

    probs = predict(files)
    y_true = np.array([class_indices[p.parent.name] for p in files])
    y_pred = probs.argmax(axis=1)
    confidence = probs.max(axis=1) * 100
    print(f"\n{split.upper()}: n={len(files)}, accuracy={(y_true == y_pred).mean():.1%}, "
          f"mean confidence={confidence.mean():.1f}%, "
          f"confidence below {THRESHOLD:.0f}%={(confidence < THRESHOLD).mean():.1%}")
    print(f"{'class':36s}{'n':>6s}{'accuracy':>11s}{'mean conf':>12s}  common wrong predictions")
    for class_name in class_names:
        mask = np.array([p.parent.name == class_name for p in files])
        if not mask.any():
            continue
        class_id = class_indices[class_name]
        accuracy = (y_pred[mask] == class_id).mean()
        mean_conf = confidence[mask].mean()
        wrong = Counter(class_names[p] for p in y_pred[mask] if p != class_id)
        top_wrong = ", ".join(f"{name} ({count})" for name, count in wrong.most_common(2)) or "-"
        print(f"{class_name:36s}{mask.sum():6d}{accuracy:10.1%}{mean_conf:11.1f}%  {top_wrong}")

try:
    from sklearn.metrics import classification_report
    files = collect_files("test")
    if files:
        probs = predict(files)
        y_true = np.array([class_indices[p.parent.name] for p in files])
        y_pred = probs.argmax(axis=1)
        print("\nTEST classification report:")
        print(classification_report(y_true, y_pred, labels=list(range(len(class_names))),
                                   target_names=class_names, zero_division=0))
except ImportError:
    print("Install scikit-learn for precision/recall/F1: pip install scikit-learn")

print("\nNote: source-specific metrics require source metadata; class folders alone do not identify dataset origin.")
