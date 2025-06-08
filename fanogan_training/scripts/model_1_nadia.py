import torch
import torch.nn as nn

# Define a Residual Block to be used in both the Generator and Discriminator
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.conv_block(x))

# Generator
class Generator(nn.Module):
    def __init__(self, opt):
        
        z_dim = opt['z_dim']
        img_channels = opt['channels']
        img_size = opt['img_size']
        
        super(Generator, self).__init__()

        # Initial dense layer to reshape the latent vector
        self.init_size = img_size // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(z_dim, 128 * self.init_size ** 2))

        # Sequence of convolutional layers and residual blocks
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            ResidualBlock(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(64),
            nn.Conv2d(64, img_channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, opt):
        
        img_channels = opt['channels']
        img_size = opt['img_size']
        
        super(Discriminator, self).__init__()

        # Sequence of convolutional layers and residual blocks
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(img_channels, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(128),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Final dense layer to produce a single output score
        ds_size = img_size // 8
        self.adv_layer = nn.Sequential(nn.Linear(256 * ds_size * ds_size, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity

