import os
import tensorflow as tf
import numpy as np

from config import IMAGE_SIZE, MODEL_PATH

DATASET_PATH = "../Datasets/Images"

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

# Build dataset with file paths
dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    image_size=IMAGE_SIZE,
    batch_size=1,
    shuffle=False,
    label_mode="binary"
)

# Build the filenames in the same sorted order TensorFlow uses
ai_dir = os.path.join(DATASET_PATH, "AI")
real_dir = os.path.join(DATASET_PATH, "REAL")

file_paths = (
    [os.path.join(ai_dir, f) for f in sorted(os.listdir(ai_dir)) if f.endswith(".jpg")] +
    [os.path.join(real_dir, f) for f in sorted(os.listdir(real_dir)) if f.endswith(".jpg")]
)

target_name = "real_7.jpg"

for (image, label), path in zip(dataset, file_paths):

    if os.path.basename(path) == target_name:

        prediction = model.predict(image, verbose=0)[0][0]

        print("\nFound:", path)
        print("True Label :", int(label.numpy().item()))
        print("Prediction :", prediction)
        print("Predicted  :", "REAL" if prediction >= 0.5 else "AI")

        break