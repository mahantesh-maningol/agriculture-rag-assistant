"""
Train one tomato classifier on diseases + nutrient deficiencies + pests.
Run prepare_dataset.py first.
"""
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

DATA = Path("dataset")
IMG = (224, 224)
BATCH = 32
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

# ---------------------------------------------------------------- data
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA / "train", image_size=IMG, batch_size=BATCH, seed=SEED)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA / "val", image_size=IMG, batch_size=BATCH, shuffle=False)
test_ds = tf.keras.utils.image_dataset_from_directory(
    DATA / "test", image_size=IMG, batch_size=BATCH, shuffle=False)

class_names = train_ds.class_names
assert class_names == val_ds.class_names == test_ds.class_names, "Class folders differ between splits"
num_classes = len(class_names)
print("Classes:", class_names)

# the app needs this to turn model output back into names
with open("class_names.json", "w") as f:
    json.dump(class_names, f)

# class weights: rare classes count more, so big classes don't dominate
counts = [len(list((DATA / "train" / c).glob("*"))) for c in class_names]
total = sum(counts)
class_weight = {i: total / (num_classes * n) for i, n in enumerate(counts)}

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# --------------------------------------------------------------- model
augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.2),
], name="augmentation")  # only active during training

base = tf.keras.applications.MobileNetV2(
    input_shape=IMG + (3,), include_top=False, weights="imagenet")
base.trainable = False

inputs = tf.keras.Input(shape=IMG + (3,))
x = augment(inputs)
x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)  # MobileNetV2 expects [-1, 1]
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                     restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint("tomato_model.keras", monitor="val_accuracy",
                                       save_best_only=True),
]

# ---------------------------------------------- phase 1: train new head
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=10,
          class_weight=class_weight, callbacks=callbacks)

# ------------------------------------- phase 2: fine-tune top of the base
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=15,
          class_weight=class_weight, callbacks=callbacks)

# ------------------------------------------------ final, honest check
print("\nTest set (never used for training or tuning):")
model.evaluate(test_ds)

try:
    from sklearn.metrics import classification_report
    y_true = np.concatenate([y.numpy() for _, y in test_ds])
    y_pred = np.argmax(model.predict(test_ds), axis=1)
    print(classification_report(y_true, y_pred, target_names=class_names))
except ImportError:
    print("pip install scikit-learn for a per-class report")

model.save("tomato_model.keras")
