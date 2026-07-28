from preprocess import load_images

# Dataset paths
AI_FOLDER = "../Datasets/Images/AI"
REAL_FOLDER = "../Datasets/Images/REAL"


def load_dataset():
    print("Loading Image Dataset...\n")

    ai_images, ai_labels = load_images(AI_FOLDER, 1)
    real_images, real_labels = load_images(REAL_FOLDER, 0)

    images = ai_images + real_images
    labels = ai_labels + real_labels

    print(f"AI Images   : {len(ai_images)}")
    print(f"Real Images : {len(real_images)}")
    print(f"Total Images: {len(images)}")

    return images, labels


if __name__ == "__main__":
    load_dataset()