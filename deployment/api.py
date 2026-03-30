# api.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import os
import torch
from torchvision.utils import save_image
import zipfile
import glob
from pathlib import Path
import shutil
import sys

# Add parent folder for generator.py
sys.path.append(str(Path(__file__).resolve().parent.parent))
from generator import Generator

# -----------------------------
# CONFIG
# -----------------------------
CHECKPOINT_DIR = r"D:\3-2SEM\GAN-FOR-IMAGES\SKILL_3\checkpoints"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated")
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 100
img_channels = 3

# -----------------------------
# LOAD LATEST GENERATOR
# -----------------------------
all_ckpts = glob.glob(os.path.join(CHECKPOINT_DIR, "G*.pth"))
if not all_ckpts:
    raise FileNotFoundError(f"No checkpoint found in {CHECKPOINT_DIR}!")

MODEL_PATH = max(all_ckpts, key=os.path.getmtime)

generator = Generator(latent_dim=latent_dim, img_channels=img_channels).to(device)
generator.load_state_dict(torch.load(MODEL_PATH, map_location=device))
generator.eval()

# -----------------------------
# FASTAPI APP
# -----------------------------
app = FastAPI(title="Leaf GAN Image Generator API")

class GenerateRequest(BaseModel):
    n_images: Optional[int] = 5

@app.get("/")
def home():
    return {"message": "Leaf GAN Image Generator API is running"}

@app.post("/generate/")
def generate_images_api(request: GenerateRequest):

    n_images = request.n_images

    if n_images < 1 or n_images > 20:
        raise HTTPException(status_code=400, detail="n_images must be between 1 and 20")

    try:
        # Clean output folder
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        generated_paths = []

        with torch.no_grad():
            for i in range(n_images):
                z = torch.randn(1, latent_dim, device=device)
                fake_img = generator(z)
                fake_img = (fake_img + 1) / 2.0

                img_path = os.path.join(OUTPUT_DIR, f"leaf_{i+1}.png")
                save_image(fake_img, img_path)
                generated_paths.append(img_path)

        # Create ZIP
        zip_path = os.path.join(OUTPUT_DIR, "leaves.zip")
        with zipfile.ZipFile(zip_path, "w") as zipf:
            for file in generated_paths:
                zipf.write(file, os.path.basename(file))

        # Return actual ZIP file
        return FileResponse(
            path=zip_path,
            filename="leaves.zip",
            media_type="application/zip"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))