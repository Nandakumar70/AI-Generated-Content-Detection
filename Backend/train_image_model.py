import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from Backend.models.clip_model import CLIPFakeDetector

# Paths
DATASET_PATH = "Datasets/Images"
MODEL_SAVE_PATH = "Backend/models/clip_model.pt"

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting fast-training on: {device.type.upper()}")

    # 1. Load and prep the data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])

    dataset = datasets.ImageFolder(root=DATASET_PATH, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    print(f"📂 Found {len(dataset)} images in {DATASET_PATH}.")
    print(f"🏷️ Class mapping forced to: {dataset.class_to_idx} (0 = AI, 1 = REAL)")

    # 2. Load model
    model = CLIPFakeDetector().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 3. Fast Training Loop
    model.train()
    for epoch in range(3):
        total_loss = 0
        correct = 0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds == labels).sum().item()
            
        acc = 100. * correct / len(dataset)
        print(f"⭐ Epoch {epoch+1}/3 - Loss: {total_loss/len(dataloader):.4f} - Accuracy: {acc:.2f}%")

    # 4. Save the fixed weights
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ FIXED model successfully saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_model()