import os
import cv2
import numpy as np

IMAGE_SIZE = (128, 128)

def load_images(folder, label):
    images = []
    labels = []

    files = os.listdir(folder)

    print(f"\nLoading images from: {folder}")
    print(f"Total images found: {len(files)}\n")

    count = 0

    for file in files:
        path = os.path.join(folder, file)

        img = cv2.imread(path)

        if img is None:
            continue

        img = cv2.resize(img, IMAGE_SIZE)
        img = img.astype(np.float32) / 255.0

        images.append(img)
        labels.append(label)

        count += 1

        if count % 1000 == 0:
            print(f"{count}/{len(files)} images loaded...")

    print(f"Finished loading {count} images from {folder}\n")

    return images, labels