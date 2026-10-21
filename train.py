import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2

# =========================
# CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parent            # folder that contains train.py
DATASET_PATH = BASE_DIR / "dataset"           # output of prepare_dataset.py (train/ val/ test/)
SAVE_DIR = BASE_DIR / "backend" / "saved_model"       # same folder predict_service.py reads
MODEL_SAVE_PATH = SAVE_DIR / "model.keras"   # native format; legacy .h5 fails with Keras 3 here
CLASSES_PATH = SAVE_DIR / "classes.json"

IMG_SIZE = (320, 320)     # was 224; AgriNet leaves are small in a ~2000 px frame
MAX_CLASS_WEIGHT = 10.0   # lower this (e.g. 3-5) if healthy leaves get flagged as deficient too often
BATCH_SIZE = 32
HEAD_EPOCHS = 10          # phase 1: train only the new classifier head
FINE_TUNE_EPOCHS = 30     # phase 2: also fine-tune the top of MobileNetV2 (early stopping ends it sooner)
FINE_TUNE_LAYERS = 100    # MobileNetV2 has ~154 layers; AgriNet needs more than the last 30 adapted
PATIENCE = 5
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

# =========================
# LOAD DATA (separate train / val / test folders, no validation_split)
# =========================
def load(split: str, shuffle: bool):
    return tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH / split,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=shuffle,
        seed=SEED,
    )

train_data = load("train", shuffle=True)
val_data = load("val", shuffle=False)
test_data = load("test", shuffle=False)

class_names = train_data.class_names
assert class_names == val_data.class_names == test_data.class_names, \
    "Class folders differ between train/val/test"
num_classes = len(class_names)

# same {"ClassName": index} format your predict service already expects
class_indices = {name: i for i, name in enumerate(class_names)}
print("Classes:", class_indices)

# Rare classes count more, so big classes don't dominate. Clipped at 10 so the
# tiniest classes can't destabilise training.
counts = [len(list((DATASET_PATH / "train" / c).glob("*"))) for c in class_names]
total = sum(counts)
class_weight = {i: min(total / (num_classes * n), MAX_CLASS_WEIGHT) for i, n in enumerate(counts)}
for name, n in zip(class_names, counts):
    print(f"{name:32s} train images: {n:5d}")

weights_tensor = tf.constant([class_weight[i] for i in range(num_classes)], dtype=tf.float32)


def add_weights(x, y):
    return x, y, tf.gather(weights_tensor, y)


# Weight BOTH training and validation. If only training were weighted, early
# stopping would watch a validation loss dominated by the big PlantVillage classes
# and stop before the AgriNet classes had learned anything.
train_fit = train_data.map(add_weights, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
val_fit = val_data.map(add_weights, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
test_data = test_data.prefetch(AUTOTUNE)

# =========================
# BUILD MODEL
# =========================
# Augmentation and scaling live INSIDE the model, so predict_service.py
# only has to pass raw 0-255 pixels.
augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    # No brightness/contrast jitter on purpose: colour is the main clue for
    # nitrogen and potassium deficiency.
], name="augmentation")   # only active during training

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=IMG_SIZE + (3,),
)
base_model.trainable = False   # phase 1: keep pretrained weights intact

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = augmentation(inputs)
x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)   # MobileNetV2 expects [-1, 1]
x = base_model(x, training=False)                 # keeps BatchNorm in inference mode
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)


def callbacks():
    return [tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=PATIENCE, restore_best_weights=True)]


# =========================
# PHASE 1: TRAIN THE NEW HEAD
# =========================
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(train_fit, validation_data=val_fit, epochs=HEAD_EPOCHS,
          callbacks=callbacks())

# =========================
# PHASE 2: FINE-TUNE TOP LAYERS (low learning rate)
# =========================
base_model.trainable = True
for layer in base_model.layers[:-FINE_TUNE_LAYERS]:
    layer.trainable = False
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

model.compile(   # must recompile after changing trainable flags
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
history = model.fit(train_fit, validation_data=val_fit, epochs=FINE_TUNE_EPOCHS,
                    callbacks=callbacks())

# =========================
# SAVE MODEL + CLASSES FIRST, so a later error can never lose the trained model
# =========================
SAVE_DIR.mkdir(parents=True, exist_ok=True)
model.save(MODEL_SAVE_PATH)
with open(CLASSES_PATH, "w") as f:
    json.dump(class_indices, f, indent=2)
print(f"✅ Model saved at: {MODEL_SAVE_PATH}")

# =========================
# EVALUATE ON THE UNTOUCHED TEST SET
# =========================
print("\nTest set (never used for training or tuning):")
model.evaluate(test_data)

try:
    from sklearn.metrics import classification_report
    y_true = np.concatenate([y.numpy() for _, y in test_data])
    y_pred = np.argmax(model.predict(test_data), axis=1)
    print(classification_report(y_true, y_pred, target_names=class_names))
except ImportError:
    print("pip install scikit-learn to see a per-class report")

print(f"✅ Done. Model: {MODEL_SAVE_PATH}  Classes: {CLASSES_PATH}")