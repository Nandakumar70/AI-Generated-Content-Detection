import torch
import torch.nn as nn
from transformers import CLIPVisionModel
import os

class CLIPFakeDetector(nn.Module):
    def __init__(self):
        super(CLIPFakeDetector, self).__init__()
        # use_safetensors bypasses the PyTorch security block
        self.clip = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True)
        self.classifier = nn.Sequential(
            nn.Linear(self.clip.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1) 
        )

# Create the models folder if it doesn't exist
os.makedirs("Backend/models", exist_ok=True)

# Initialize untrained model and save it
print("Generating pipeline demo model...")
model = CLIPFakeDetector()
torch.save(model.state_dict(), "Backend/models/clip_model.pt")
print("Success! Placeholder clip_model.pt saved. You can now start your API.")