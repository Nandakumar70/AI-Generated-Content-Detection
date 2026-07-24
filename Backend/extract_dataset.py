import os
import io
import pandas as pd
from PIL import Image

TRAIN_FILE = "../Datasets/Raw/CIFAKE/train-00000-of-00001.parquet"

AI_FOLDER = "../Datasets/Images/AI"
REAL_FOLDER = "../Datasets/Images/REAL"

os.makedirs(AI_FOLDER, exist_ok=True)
os.makedirs(REAL_FOLDER, exist_ok=True)


def extract_dataset(file_path):
    print(f"Reading {file_path}")

    df = pd.read_parquet(file_path)

    ai_count = 0
    real_count = 0

    for index, row in df.iterrows():

        image_dict = row["image"]
        label = row["label"]

        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_dict["bytes"]))

        if label == 1:
            image.save(os.path.join(AI_FOLDER, f"ai_{ai_count}.jpg"))
            ai_count += 1
        else:
            image.save(os.path.join(REAL_FOLDER, f"real_{real_count}.jpg"))
            real_count += 1

        if (index + 1) % 1000 == 0:
            print(f"{index + 1} images processed...")

    print("\nFinished!")
    print(f"AI Images   : {ai_count}")
    print(f"Real Images : {real_count}")


if __name__ == "__main__":
    extract_dataset(TRAIN_FILE)