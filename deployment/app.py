# app.py
import os
import glob
import torch
from torchvision.utils import save_image
import streamlit as st
import zipfile
from pathlib import Path
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
DISPLAY_SIZE = 110

st.set_page_config(
    page_title="Leaf AI Generator",
    page_icon="🌿",
    layout="wide"
)

# -----------------------------
# PREMIUM CSS
# -----------------------------
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
}

/* Hero Title */
.hero {
    text-align: center;
    font-size: 48px;
    font-weight: 800;
    color: white;
    margin-top: 10px;
}

.subtext {
    text-align: center;
    font-size: 18px;
    color: #cfcfcf;
    margin-bottom: 30px;
}

/* Glass Card */
.glass {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(12px);
    border-radius: 20px;
    padding: 30px;
    margin-top: 20px;
}

/* Image hover */
img {
    border-radius: 15px;
    transition: transform 0.3s ease;
}
img:hover {
    transform: scale(1.15);
}

/* Button styling */
.stButton>button {
    width: 100%;
    height: 55px;
    border-radius: 12px;
    font-size: 18px;
    font-weight: bold;
    background: linear-gradient(to right, #00c6ff, #0072ff);
    color: white;
    border: none;
}
.stButton>button:hover {
    background: linear-gradient(to right, #00ffae, #0072ff);
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
all_ckpts = glob.glob(os.path.join(CHECKPOINT_DIR, "G*.pth"))
if not all_ckpts:
    st.error("No checkpoint found!")
    st.stop()

MODEL_PATH = max(all_ckpts, key=os.path.getmtime)

generator = Generator(latent_dim=latent_dim, img_channels=img_channels).to(device)
generator.load_state_dict(torch.load(MODEL_PATH, map_location=device))
generator.eval()

# -----------------------------
# HERO SECTION
# -----------------------------
st.markdown('<div class="hero">🌿 Leaf AI Image Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="subtext">Generate realistic synthetic leaf images using GAN</div>', unsafe_allow_html=True)

# -----------------------------
# SIDEBAR CONTROL PANEL
# -----------------------------
with st.sidebar:
    st.header("⚙️ Controls")
    num_images = st.slider("Number of Images", 1, 20, 6)
    generate_clicked = st.button("🚀 Generate")

# -----------------------------
# GENERATION SECTION
# -----------------------------
if generate_clicked:

    for f in os.listdir(OUTPUT_DIR):
        os.remove(os.path.join(OUTPUT_DIR, f))

    generated_paths = []

    with st.spinner("Generating AI images..."):
        with torch.no_grad():
            for i in range(num_images):
                z = torch.randn(1, latent_dim, device=device)
                fake_img = generator(z)
                fake_img = (fake_img + 1) / 2.0

                img_path = os.path.join(OUTPUT_DIR, f"leaf_{i+1}.png")
                save_image(fake_img, img_path)
                generated_paths.append(img_path)

    # Output section
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("## 🌟 Generated Images")

    for i in range(0, len(generated_paths), 5):
        cols = st.columns(5)
        for j in range(5):
            if i + j < len(generated_paths):
                cols[j].image(generated_paths[i + j], width=DISPLAY_SIZE)

    # ZIP
    zip_path = os.path.join(OUTPUT_DIR, "leaves.zip")
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for file in generated_paths:
            zipf.write(file, os.path.basename(file))

    with open(zip_path, "rb") as f:
        st.download_button(
            label="📥 Download All Images",
            data=f,
            file_name="leaves.zip",
            mime="application/zip"
        )

    st.success(f"✨ {num_images} images generated successfully!")

    st.markdown('</div>', unsafe_allow_html=True)