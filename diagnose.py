"""
Compare how the trained model does on PlantVillage (pv_) vs AgriNet (ag_) images.
Run from anywhere:  python diagnose.py
"""
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
DATA = BASE_DIR / "dataset"
MODEL_PATH = BASE_DIR / "backend" / "saved_model" / "model.keras"
CLASSES_PATH = BASE_DIR / "backend" / "saved_model" / "classes.json"

IMG_SIZE = (224, 224)
THRESHOLD = 70.0        # same as predict_service.py
MAX_PER_GROUP = 300     # keeps the run fast
SEED = 42

model = tf.keras.models.load_model(MODEL_PATH)
IMG_SIZE = tuple(model.input_shape[1:3])   # always matches whatever train.py used
with open(CLASSES_PATH) as f:
    class_indices = json.load(f)
class_names = [n for n, _ in sorted(class_indices.items(), key=lambda kv: kv[1])]


def load(path):
    """Same decode + resize as image_dataset_from_directory used in training."""
    img = tf.io.decode_image(tf.io.read_file(str(path)), channels=3, expand_animations=False)
    return tf.image.resize(img, IMG_SIZE).numpy()


def predict(paths):
    out = []
    for i in range(0, len(paths), 32):
        batch = np.stack([load(p) for p in paths[i:i + 32]])
        out.append(model(batch, training=False).numpy())
    return np.concatenate(out)


def files_for(split, prefix):
    return [p for p in (DATA / split).glob("*/*") if p.name.startswith(prefix)]


# Classes that only exist in PlantVillage (no AgriNet images in train)
ag_classes = {p.parent.name for p in files_for("train", "ag_")}
pv_only = set(class_names) - ag_classes

rng = random.Random(SEED)
ag_test = None

print(f"{'split':6s}{'source':14s}{'n':>5s}{'accuracy':>10s}{'mean conf':>11s}{'conf<' + str(int(THRESHOLD)) + '%':>10s}")
for split in ("train", "val", "test"):
    for source, prefix in (("PlantVillage", "pv_"), ("AgriNet", "ag_")):
        files = files_for(split, prefix)
        if not files:
            continue
        rng.shuffle(files)
        files = files[:MAX_PER_GROUP]
        probs = predict(files)
        pred = probs.argmax(1)
        conf = probs.max(1) * 100
        true = np.array([class_names.index(p.parent.name) for p in files])
        correct = pred == true
        print(f"{split:6s}{source:14s}{len(files):5d}{correct.mean():10.1%}"
              f"{conf.mean():10.1f}%{(conf < THRESHOLD).mean():10.1%}")
        if split == "test" and source == "AgriNet":
            ag_test = (files, true, pred, conf)

if ag_test:
    files, true, pred, conf = ag_test
    print("\nAgriNet TEST images, per class")
    print(f"{'true class':32s}{'n':>5s}{'acc':>8s}{'mean conf':>11s}   most common wrong prediction")
    for ci, name in enumerate(class_names):
        mask = true == ci
        if not mask.any():
            continue
        acc = (pred[mask] == ci).mean()
        wrong = Counter(class_names[p] for p in pred[mask] if p != ci)
        top_wrong = ", ".join(f"{k} ({v})" for k, v in wrong.most_common(2)) or "-"
        print(f"{name:32s}{mask.sum():5d}{acc:8.1%}{conf[mask].mean():10.1f}%   {top_wrong}")

    to_pv_only = np.mean([class_names[p] in pv_only for p in pred])
    print(f"\nAgriNet test images predicted as a PlantVillage-only class: {to_pv_only:.1%}")