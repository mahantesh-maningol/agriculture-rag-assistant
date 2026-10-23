"""Train the tomato classification model from dataset/{train,val,test}/<class>/.

PlantVillage tomato images and Tomato Village Variant-A images can be mixed in
these class folders. No filename prefixes are required. The dataset must already
contain only the intended tomato images and must use the exact class folder names
listed in CLASS_NAMES.
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2

# =========================
# CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset"
SAVE_DIR = BASE_DIR / "backend" / "saved_model"
MODEL_SAVE_PATH = SAVE_DIR / "model.keras"
CLASSES_PATH = SAVE_DIR / "classes.json"

# Keep this order synchronized with the prediction service and product tool.
CLASS_NAMES = [
    "Bacterial_spot",
    "Early_blight",
    "Healthy",
    "Jassids_and_mites",
    "Late_blight",
    "Leaf_Mold",
    "Leaf_miner",
    "Magnesium_deficiency",
    "Mites",
    "Nitrogen_deficiency",
    "Pottassium_deficiency",  # Preserve the existing spelling used by your API.
    "Septoria_leaf_spot",
    "Spider_mites",
    "Target_Spot",
    "Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_mosaic_virus",
]

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
MAX_CLASS_WEIGHT = 10.0
HEAD_EPOCHS = 5
FINE_TUNE_EPOCHS = 10
FINE_TUNE_LAYERS = 100
PATIENCE = 5
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_split(split):
    """Collect images from class folders without requiring filename prefixes."""
    split_dir = DATASET_PATH / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing dataset split directory: {split_dir}")

    records = []
    found_classes = set()
    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        if class_dir.name not in CLASS_NAMES:
            raise ValueError(
                f"Unexpected class folder '{class_dir.name}' in {split_dir}. "
                "Rename/map it to one of CLASS_NAMES or remove it from this dataset."
            )
        for path in class_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                records.append((str(path), class_dir.name))
                found_classes.add(class_dir.name)

    if not records:
        raise ValueError(f"No supported images found in {split_dir}")

    missing = set(CLASS_NAMES) - found_classes
    if missing:
        print(f"WARNING: {split} has no images for: {', '.join(sorted(missing))}")

    counts = Counter(name for _, name in records)
    print(f"{split}: {len(records)} images across {len(counts)} classes")
    return records, counts


def decode_image(path, label):
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMG_SIZE)
    return image, label


def make_dataset(records, shuffle=False):
    class_indices = {name: i for i, name in enumerate(CLASS_NAMES)}
    paths = [path for path, _ in records]
    labels = [class_indices[name] for _, name in records]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(records), seed=SEED, reshuffle_each_iteration=True)
    return ds.map(decode_image, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)


# =========================
# LOAD EXISTING TRAIN / VAL / TEST SPLITS
# =========================
train_records, train_counts = collect_split("train")
val_records, _ = collect_split("val")
test_records, _ = collect_split("test")

class_indices = {name: i for i, name in enumerate(CLASS_NAMES)}
num_classes = len(CLASS_NAMES)
print("Class mapping:", json.dumps(class_indices, indent=2))

train_data = make_dataset(train_records, shuffle=True)
val_data = make_dataset(val_records)
test_data = make_dataset(test_records)

# Compute class weights using training images only.
total = len(train_records)
class_weight = {}
for i, name in enumerate(CLASS_NAMES):
    count = train_counts.get(name, 0)
    if count == 0:
        # No examples means the class cannot be learned from this training set.
        continue
    class_weight[i] = min(total / (num_classes * count), MAX_CLASS_WEIGHT)
    print(f"{name:40s} train images: {count:5d} class_weight: {class_weight[i]:.2f}")

# Use sample weights in training. Validation remains unweighted so val_loss/val_accuracy
# reflect the actual validation distribution rather than the training reweighting.
weights_tensor = tf.constant(
    [class_weight.get(i, 1.0) for i in range(num_classes)], dtype=tf.float32
)

def add_weights(x, y):
    return x, y, tf.gather(weights_tensor, y)

train_fit = train_data.map(add_weights, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

# =========================
# BUILD MODEL
# =========================
augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
], name="augmentation")

base_model = MobileNetV2(
    weights="imagenet", include_top=False, input_shape=IMG_SIZE + (3,)
)
base_model.trainable = False

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = augmentation(inputs)
x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)


def callbacks():
    return [tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=PATIENCE, restore_best_weights=True
    )]


# =========================
# PHASE 1: TRAIN CLASSIFIER HEAD
# =========================
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(train_fit, validation_data=val_data, epochs=HEAD_EPOCHS, callbacks=callbacks())

# =========================
# PHASE 2: FINE-TUNE TOP LAYERS
# =========================
base_model.trainable = True
for layer in base_model.layers[:-FINE_TUNE_LAYERS]:
    layer.trainable = False
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(train_fit, validation_data=val_data, epochs=FINE_TUNE_EPOCHS, callbacks=callbacks())

# =========================
# SAVE MODEL AND FIXED CLASS MAPPING
# =========================
SAVE_DIR.mkdir(parents=True, exist_ok=True)
model.save(MODEL_SAVE_PATH)
with open(CLASSES_PATH, "w", encoding="utf-8") as f:
    json.dump(class_indices, f, indent=2)
print(f"Model saved: {MODEL_SAVE_PATH}")
print(f"Class mapping saved: {CLASSES_PATH}")

# =========================
# FINAL HELD-OUT TEST EVALUATION
# =========================
print("\nCombined held-out test set:")
model.evaluate(test_data)
try:
    from sklearn.metrics import classification_report
    y_true = np.concatenate([y.numpy() for _, y in test_data])
    y_prob = model.predict(test_data)
    y_pred = np.argmax(y_prob, axis=1)
    print(classification_report(
        y_true, y_pred, labels=list(range(num_classes)),
        target_names=CLASS_NAMES, zero_division=0
    ))
except ImportError:
    print("Install scikit-learn for per-class metrics: pip install scikit-learn")

print(f"Done. Model: {MODEL_SAVE_PATH} | Classes: {CLASSES_PATH}")
