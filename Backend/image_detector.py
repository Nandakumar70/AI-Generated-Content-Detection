from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
import torch
from torchvision import transforms
from PIL import Image
import io
import os

# Import your new CLIP model architecture
from models.clip_model import CLIPFakeDetector

router = APIRouter()

# 1. Initialize Hardware and Model State
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading CLIP model on: {device.type.upper()}")

model = CLIPFakeDetector().to(device)
MODEL_PATH = "../Models/clip_model.pt"

# 2. Load Weights and Set to Evaluation Mode
if os.path.exists(MODEL_PATH):
    # map_location ensures it loads correctly whether on GPU or CPU
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()  # Set the model to inference mode (disables dropout, etc.)
    print("✅ CLIP Model weights loaded successfully.")
else:
    print("⚠️ Warning: clip_model.pt not found. Please run the training script first.")

# 3. Exact Same Transforms Used During Training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
])

class DetectionResponse(BaseModel):
    filename: str
    prediction: str
    confidence_score: float

@router.post("/detect/image", response_model=DetectionResponse)
async def detect_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File uploaded is not an image.")

    try:
        # Read and preprocess the image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Apply transforms and add batch dimension (B, C, H, W)
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Run inference without tracking gradients to save memory and increase speed
        with torch.no_grad():
            output = model(input_tensor)
            probability = output.item() # Get the raw float value

        # 0 = AI-Generated (FAKE), 1 = Real Photograph
        is_real = probability > 0.5
        label = "Real Photograph" if is_real else "AI-Generated"
        
        # Calculate confidence percentage
        confidence = probability if is_real else (1.0 - probability)

        return DetectionResponse(
            filename=file.filename,
            prediction=label,
            confidence_score=round(confidence * 100, 2)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))