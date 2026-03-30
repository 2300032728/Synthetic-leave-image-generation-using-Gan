import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, img_channels=3, feature_maps=64):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            # Input: 64x64x3
            nn.Conv2d(img_channels, feature_maps, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_maps, feature_maps*2, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(feature_maps*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_maps*2, feature_maps*4, kernel_size=4, stride=2, padding=1),  # 8x8
            nn.BatchNorm2d(feature_maps*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_maps*4, feature_maps*8, kernel_size=4, stride=2, padding=1),  # 4x4
            nn.BatchNorm2d(feature_maps*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(feature_maps*8*4*4, 1),
            nn.Sigmoid()  # Probability of "real"
        )

    def forward(self, x):
        return self.net(x)
