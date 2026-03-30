import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader

from SKILL_3.generator import Generator
from discriminator import Discriminator

# ================== CONFIG ==================
EPOCHS = 2000
BATCH_SIZE = 64
LATENT_DIM = 100
IMAGE_SIZE = 64
CHANNELS = 3
LR = 0.0002
BETA1 = 0.5

DATA_PATH = "preprocessed/Lettuce_disease_datasets"
SAMPLE_DIR = "samples"
CHECKPOINT_DIR = "checkpoints"

os.makedirs(SAMPLE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ================== DATA LOADER ==================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
dataset = datasets.ImageFolder(root=DATA_PATH, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ================== MODELS ==================
G = Generator(latent_dim=LATENT_DIM).to(device)
D = Discriminator().to(device)

# ================== LOSS & OPTIMIZERS ==================
criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=LR, betas=(BETA1, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=LR, betas=(BETA1, 0.999))

# ================== FIXED NOISE (16 images) ==================
fixed_noise = torch.randn(16, LATENT_DIM, device=device)

# ================== TRAINING LOOP ==================
for epoch in range(1, EPOCHS + 1):
    for i, (real_imgs, _) in enumerate(dataloader):
        real_imgs = real_imgs.to(device)
        batch_size = real_imgs.size(0)

        # -------- LABEL SMOOTHING + STABILITY --------
        real_labels = torch.full((batch_size, 1), 0.9, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)

        if torch.rand(1).item() < 0.05:
            real_labels, fake_labels = fake_labels, real_labels

        real_imgs = real_imgs + 0.05 * torch.randn_like(real_imgs)

        # -------- TRAIN DISCRIMINATOR --------
        optimizer_D.zero_grad()

        output_real = D(real_imgs)
        d_loss_real = criterion(output_real, real_labels)

        noise = torch.randn(batch_size, LATENT_DIM, device=device)
        fake_imgs = G(noise)

        output_fake = D(fake_imgs.detach())
        d_loss_fake = criterion(output_fake, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(D.parameters(), 1.0)
        optimizer_D.step()

        # -------- TRAIN GENERATOR --------
        optimizer_G.zero_grad()

        gen_labels = torch.ones(batch_size, 1, device=device)
        output = D(fake_imgs)
        g_loss = criterion(output, gen_labels)

        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
        optimizer_G.step()

        # -------- PRINT PROGRESS --------
        if (i + 1) % 50 == 0 or (i + 1) == len(dataloader):
            print(f"Epoch [{epoch}/{EPOCHS}] Batch [{i+1}/{len(dataloader)}] "
                  f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

    # -------- SAVE GRID EVERY 20 EPOCHS --------
    if epoch % 200 == 0 or epoch == 1:
        with torch.no_grad():
            fake_samples = G(fixed_noise).detach().cpu()
            fake_samples = (fake_samples + 1) / 2

        utils.save_image(
            fake_samples,
            f"{SAMPLE_DIR}/epoch_{epoch:03d}.png",
            nrow=4
        )
        print(f"✅ Grid image saved for epoch {epoch}")

# ================== SAVE MODELS ==================
torch.save(G.state_dict(), f"{CHECKPOINT_DIR}/G_final.pth")
torch.save(D.state_dict(), f"{CHECKPOINT_DIR}/D_final.pth")
print("✅ Training complete. Models saved.")
