import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_channels=3, feature_maps=64):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim

        self.net = nn.Sequential(
            # Fully connected to 4x4x512
            nn.Linear(latent_dim, 4*4*512),
            nn.BatchNorm1d(4*4*512),
            nn.ReLU(True),

            # Reshape to (batch, 512, 4, 4)
            nn.Unflatten(1, (512, 4, 4)),

            # Upsample 4x4 -> 8x8
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # 8x8 -> 16x16
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # 16x16 -> 32x32
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 32x32 -> 64x64
            nn.ConvTranspose2d(64, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output in [-1,1]
        )

    def forward(self, z):
        return self.net(z)
