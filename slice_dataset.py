import os
from datasets import load_dataset
from PIL import Image

# Point this to your existing AI folder
save_dir = "Datasets/Images/AI"
os.makedirs(save_dir, exist_ok=True)

print("Connecting to Hugging Face Stream...")

try:
    # streaming=True prevents downloading the entire dataset at once
    dataset = load_dataset("Rajarshi-Roy-research/Defactify_Image_Dataset", split="train", streaming=True)
    
    print("Successfully connected! Slicing 500 modern AI images (DALL-E 3, Midjourney v6, SD3)...")
    
    count = 0
    for item in dataset:
        # Stop once we have exactly 500 AI images
        if count >= 500:
            break
            
        # Label_A = 1 means AI-Generated (0 means Real)
        if item['Label_A'] == 1:
            img = item['Image']
            
            # Convert to RGB to prevent transparency errors
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            img.save(os.path.join(save_dir, f"modern_sota_ai_{count}.jpg"))
            count += 1
            
            # Print progress so you know it's working
            if count % 100 == 0:
                print(f"Downloaded {count}/500 images...")

    print(f"Slice complete! 500 new modern AI images added to {save_dir}.")
    
except Exception as e:
    print(f"An error occurred: {e}")