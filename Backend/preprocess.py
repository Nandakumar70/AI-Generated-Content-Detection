import os
import cv2
import numpy as np

IMAGE_SIZE = (224, 224)


def load_images(folder, label):
    images = []
    labels = []

    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        img = cv2.imread(path)

        if img is None:
            continue

        img = cv2.resize(img, IMAGE_SIZE)

        img = img / 255.0

        images.append(img)
        labels.append(label)

    return images, labels