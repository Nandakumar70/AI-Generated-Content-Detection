import tensorflow as tf

from models.cnn_model import build_model

from config import (
    IMAGE_SIZE,
    BATCH_SIZE,
    EPOCHS,
    MODEL_PATH
)

DATASET_PATH = "../Datasets/Images"

print("\nLoading Dataset...\n")

train_dataset = tf.keras.utils.image_dataset_from_directory(

    DATASET_PATH,

    validation_split=0.2,

    subset="training",

    seed=42,

    image_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    label_mode="binary"

)

validation_dataset = tf.keras.utils.image_dataset_from_directory(

    DATASET_PATH,

    validation_split=0.2,

    subset="validation",

    seed=42,

    image_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    label_mode="binary"

)

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

print("\nDataset Loaded Successfully!\n")

print("\nBuilding CNN Model...\n")

model = build_model()

model.summary()

print("\nTraining Started...\n")

history = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=EPOCHS

)

print("\nSaving Model...\n")

model.save(MODEL_PATH)

print("\nModel Saved Successfully!")

print("\nTraining Completed!")