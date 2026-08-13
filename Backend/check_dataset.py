import tensorflow as tf
import numpy as np

from config import IMAGE_SIZE, BATCH_SIZE

DATASET_PATH = "../Datasets/Images"

dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

print("Class names:", dataset.class_names)

all_labels = []

for _, labels in dataset:
    all_labels.extend(labels.numpy().flatten())

all_labels = np.array(all_labels)

print("\nTotal images :", len(all_labels))
print("AI images    :", np.sum(all_labels == 0))
print("REAL images  :", np.sum(all_labels == 1))