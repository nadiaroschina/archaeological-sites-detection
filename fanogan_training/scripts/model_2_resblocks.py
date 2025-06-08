import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, upsample=False, downsample=False, normalize=False):
        super().__init__()
        self.actv = nn.LeakyReLU(0.2)
        self.upsample = upsample
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self.normalize = normalize

        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        x = self.conv1(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        out = self._residual(x)
        out = (out + self._shortcut(x)) / math.sqrt(2)
        return out
    

class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        
        latent_dim = opt['latent_dim']
        img_size = opt['img_size']
        num_channels = opt['channels']
        
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.num_channels = num_channels

        # Compute the size of the intermediate feature maps
        num_features = img_size // 16  # Assuming 4 residual blocks with upsampling by 2x each

        self.fc = nn.Linear(latent_dim, num_features * num_features * 64)
        
        self.block1 = ResBlk(64, 64, upsample=True, normalize=True)
        self.block2 = ResBlk(64, 32, upsample=True, normalize=True)
        self.block3 = ResBlk(32, 16, upsample=True, normalize=True)
        self.block4 = ResBlk(16, num_channels, upsample=True, normalize=True)
        
        self.final_conv = nn.Conv2d(num_channels, num_channels, 3, 1, 1)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 64, self.img_size // 16, self.img_size // 16)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.final_conv(x)
        x = torch.tanh(x)
        return x

    
class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        
        latent_dim = opt['latent_dim']
        img_size = opt['img_size']
        num_channels = opt['channels']
        
        self.img_size = img_size
        self.num_channels = num_channels

        self.block1 = ResBlk(num_channels, 16, downsample=True, normalize=True)
        self.block2 = ResBlk(16, 32, downsample=True, normalize=True)
        self.block3 = ResBlk(32, 64, downsample=True, normalize=True)
        self.block4 = ResBlk(64, 128, downsample=True, normalize=True)

        self.fc = nn.Linear(128 * (img_size // 16) * (img_size // 16), 1)

    def forward(self, x):
        # Residual blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        # Flatten and classify
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        
        return torch.sigmoid(x)  # Output probability in the range [0, 1]
    
    def forward_features(self, x):
        # Residual blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.view(x.shape[0], -1)
        return x
    
    
class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        
        latent_dim = opt['latent_dim']
        img_size = opt['img_size']
        num_channels = opt['channels']
        
        self.img_size = img_size
        self.num_channels = num_channels
        self.latent_dim = latent_dim

        self.block1 = ResBlk(num_channels, 16, downsample=True, normalize=True)
        self.block2 = ResBlk(16, 32, downsample=True, normalize=True)
        self.block3 = ResBlk(32, 64, downsample=True, normalize=True)
        self.block4 = ResBlk(64, 128, downsample=True, normalize=True)

        self.adv_layer = nn.Sequential(
            nn.Linear(128 * (img_size // 16) * (img_size // 16), latent_dim),
            nn.Tanh()
        )

    def forward(self, x):
        # Residual blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        # Flatten and classify
        x = x.view(x.size(0), -1)
        validity = self.adv_layer(x)
        return validity
