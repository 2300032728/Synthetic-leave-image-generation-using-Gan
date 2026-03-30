import torch
import torch.nn as nn
from SKILL_3.generator import Generator
from discriminator import Discriminator

class DCGAN:
    def __init__(self, latent_dim=100, lr=0.0002, beta1=0.5, beta2=0.999, device='cuda'):
        self.device = device
        self.latent_dim = latent_dim

        # Initialize Generator and Discriminator
        self.G = Generator(latent_dim=latent_dim).to(device)
        self.D = Discriminator().to(device)

        # Loss function
        self.criterion = nn.BCELoss()

        # Optimizers
        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=lr, betas=(beta1, beta2))
        self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=lr, betas=(beta1, beta2))

    def generate_noise(self, batch_size):
        return torch.randn(batch_size, self.latent_dim, device=self.device)
