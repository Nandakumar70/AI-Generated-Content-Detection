import sys
import os
import torch
from torchvision import transforms
from PIL import Image
from Backend.models.clip_model import CLIPFakeDetector

def test_image(image_path):
    if not os.path.exists(image_path):
        print(f"❌ Error: Image '{image_path}' not found.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device.type.upper()}...")

    model = CLIPFakeDetector().to(device)
    weights_path = "Backend/models/clip_model.pt"

    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        model.eval()
    else:
        print("⚠️ Error: clip_model.pt not found. Run the training script first.")
        return

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])

    try:
        img = Image.open(image_path).convert("RGB")
        input_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.sigmoid(output).item()

        # 0 = AI, 1 = REAL
        is_real = prob > 0.5
        label = "Real Photograph" if is_real else "AI-Generated"
        confidence = prob if is_real else (1.0 - prob)

        print("\n" + "="*40)
        print(f"File       : {image_path}")
        print(f"Prediction : {label}")
        print(f"Confidence : {confidence * 100:.2f}%")
        print("="*40 + "\n")

    except Exception as e:
        print(f"❌ Error processing image: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_image(sys.argv[1])
    else:
        print("Usage: python -m Backend.test_image_model <path_to_image>")