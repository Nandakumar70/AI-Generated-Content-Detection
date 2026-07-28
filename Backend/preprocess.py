import os
import cv2
import numpy as np

from config import IMAGE_SIZE


def load_images(folder, label):
    images = []
    labels = []

    for filename in os.listdir(folder):

        path = os.path.join(folder, filename)

        image = cv2.imread(path)

        if image is None:
            continue

        image = cv2.resize(image, IMAGE_SIZE)

        image = image.astype("float32") / 255.0

        images.append(image)
        labels.append(label)

    return images, labels