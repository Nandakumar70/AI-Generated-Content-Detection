import tensorflow as tf

from config import IMAGE_SIZE, BATCH_SIZE

DATASET_PATH = "../Datasets/Images"

dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

images, labels = next(iter(dataset))

print("Image dtype :", images.dtype)
print("Image shape :", images.shape)
print("Minimum     :", tf.reduce_min(images).numpy())
print("Maximum     :", tf.reduce_max(images).numpy())
print("Mean        :", tf.reduce_mean(images).numpy())