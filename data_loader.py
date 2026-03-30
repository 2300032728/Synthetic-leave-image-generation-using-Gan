import cv2
from pathlib import Path
from tqdm import tqdm
import shutil

# ======================
# CONFIGURATION
# ======================
RAW_DIR = Path(r"D:/3-2SEM/GAN-FOR-IMAGES/SKILL_3/leaf")  # folder with original images
PREPROCESS_DIR = Path(r"D:/3-2SEM/GAN-FOR-IMAGES/SKILL_3/preprocessed")  # folder to store preprocessed images
IMAGE_SIZE = (128, 128)  # Resize all images to 128x128
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]  # allowed image formats

# ======================
# FUNCTION TO COPY RAW IMAGES IF PREPROCESS DIR EMPTY
# ======================
def ensure_preprocessed_images():
    # Check if preprocessed folder already has images
    all_images = [p for p in PREPROCESS_DIR.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
    if len(all_images) > 0:
        return all_images

    print(f"No images found in {PREPROCESS_DIR}. Copying from {RAW_DIR}...")
    PREPROCESS_DIR.mkdir(parents=True, exist_ok=True)

    # Copy images from RAW_DIR to PREPROCESS_DIR
    for img_path in RAW_DIR.rglob("*"):
        if img_path.suffix.lower() in IMAGE_EXTENSIONS:
            relative_path = img_path.relative_to(RAW_DIR)
            dest_path = PREPROCESS_DIR / relative_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, dest_path)

    all_images = [p for p in PREPROCESS_DIR.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
    print(f"Copied {len(all_images)} images to {PREPROCESS_DIR}")
    return all_images

# ======================
# FUNCTION TO PREPROCESS IMAGES IN-PLACE (COLOR)
# ======================
def preprocess_images_in_place():
    all_images = ensure_preprocessed_images()
    if len(all_images) == 0:
        print("No images to preprocess. Please check your leaf folder.")
        return

    print(f"Preprocessing {len(all_images)} images in-place...")

    for img_path in tqdm(all_images, desc="Preprocessing images"):
        # Read image in color (default)
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"Warning: Could not read {img_path}")
            continue

        # Resize
        resized = cv2.resize(img, IMAGE_SIZE)

        # Save back to the same file (overwrite)
        cv2.imwrite(str(img_path), resized)

    print(f"All images preprocessed in-place inside {PREPROCESS_DIR}")

# ======================
# MAIN EXECUTION
# ======================
if __name__ == "__main__":
    preprocess_images_in_place()
