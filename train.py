import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import json


# =========================
# CONFIG
# =========================
DATASET_PATH = "dataset"
MODEL_SAVE_PATH = "/backend/saved_model/model.h5"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5

# =========================
# LOAD DATA
# =========================
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)



print("Classes:", train_data.class_indices)

# =========================
# LOAD PRETRAINED MODEL
# =========================
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze pretrained layers
# base_model.trainable = False

for layer in base_model.layers[-20:]:
    layer.trainable = True

# =========================
# ADD CUSTOM CLASSIFIER
# =========================
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(train_data.num_classes, activation="softmax")
])

# =========================
# COMPILE
# =========================
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# =========================
# TRAIN
# =========================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)



# =========================
# SAVE MODEL
# =========================
os.makedirs("saved_model", exist_ok=True)
model.save(MODEL_SAVE_PATH)

with open("saved_model/classes.json", "w") as f:
    json.dump(train_data.class_indices, f)

print(f"✅ Model saved at: {MODEL_SAVE_PATH}")