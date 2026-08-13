import tensorflow as tf
import numpy as np

from config import IMAGE_SIZE, MODEL_PATH

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded.")

image_path = input("\nEnter image path: ").strip()

# Load exactly like image_dataset_from_directory()
img = tf.keras.utils.load_img(
    image_path,
    target_size=IMAGE_SIZE,
    color_mode="rgb"
)

img = tf.keras.utils.img_to_array(img)

print("\nImage Info")
print("Shape :", img.shape)
print("Dtype :", img.dtype)
print("Min   :", img.min())
print("Max   :", img.max())
print("Mean  :", img.mean())

# DO NOT divide by 255
img = np.expand_dims(img, axis=0)

prediction = model.predict(img, verbose=0)[0][0]

print("\nRaw Prediction :", prediction)

if prediction >= 0.5:
    print("Prediction : REAL")
else:
    print("Prediction : AI")