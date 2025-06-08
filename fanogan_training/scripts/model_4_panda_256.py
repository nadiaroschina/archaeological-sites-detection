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
            self.norm2 = nn.InstanceNorm2d(dim_out, affine=True)
            
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
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        out = self._residual(x)
        out = (out + self._shortcut(x)) / math.sqrt(2)
        return out
    

class PandaGenerator(nn.Module):
    
    def __init__(self, opt):
        
        super(PandaGenerator, self).__init__()
        
        self.latent_dim = opt['latent_dim']
        self.img_size = opt['img_size']
        self.num_channels = opt['channels']
        self.num_features = self.img_size // 64  # size of the intermediate feature maps = 256 // 64 = 4

        self.fc = nn.Linear(self.latent_dim, self.num_features * self.num_features * 512)
        self.res_blocks = nn.Sequential(
            ResBlk(512, 512, upsample=False, normalize=True),
            ResBlk(512, 512, upsample=False, normalize=True),
            ResBlk(512, 256, upsample=True, normalize=True),
            ResBlk(256, 128, upsample=True, normalize=True),
            ResBlk(128, 64, upsample=True, normalize=True),
            ResBlk(64, 32, upsample=True, normalize=True),
            ResBlk(32, 16, upsample=True, normalize=True),
            ResBlk(16, self.num_channels, upsample=True, normalize=True)
        )
        self.final_conv = nn.Conv2d(self.num_channels, self.num_channels, 3, 1, 1)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 512, self.num_features, self.num_features)
        x = self.res_blocks(x)
        x = self.final_conv(x)
        x = torch.tanh(x)
        return x
    
class PandaDiscriminator(nn.Module):
    
    def __init__(self, opt):
        
        super(PandaDiscriminator, self).__init__()
        
        self.latent_dim = opt['latent_dim']
        self.img_size = opt['img_size']
        self.num_channels = opt['channels']

        self.res_blocks = nn.Sequential(
            ResBlk(self.num_channels, 16, downsample=True, normalize=True),
            ResBlk(16, 32, downsample=True, normalize=True),
            ResBlk(32, 64, downsample=True, normalize=True),
            ResBlk(64, 128, downsample=True, normalize=True),
            ResBlk(128, 256, downsample=True, normalize=True),
            ResBlk(256, 512, downsample=True, normalize=True),
            ResBlk(512, 512, downsample=False, normalize=True),
            ResBlk(512, 512, downsample=False, normalize=True)
        )
        self.fc = nn.Linear(512 * (self.img_size // 64) * (self.img_size // 64), 1)
        

    def forward(self, x):
        x = self.res_blocks(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return torch.sigmoid(x)
    
    def forward_features(self, x):
        x = self.res_blocks(x)
        x = x.view(x.shape[0], -1)
        return x
    
class PandaEncoder(nn.Module):
    
    def __init__(self, opt):
        
        super(ZebraEncoder, self).__init__()
        
        self.latent_dim = opt['latent_dim']
        self.img_size = opt['img_size']
        self.num_channels = opt['channels']

        self.res_blocks = nn.Sequential(
            ResBlk(self.num_channels, 16, downsample=True, normalize=True),
            ResBlk(16, 32, downsample=True, normalize=True),
            ResBlk(32, 64, downsample=True, normalize=True),
            ResBlk(64, 128, downsample=True, normalize=True),
            ResBlk(128, 256, downsample=True, normalize=True),
            ResBlk(256, 512, downsample=True, normalize=True),
            ResBlk(512, 512, downsample=False, normalize=True),
            ResBlk(512, 512, downsample=False, normalize=True)
        )
        self.adv_layer = nn.Sequential(
            nn.Linear(512 * (self.img_size // 64) * (self.img_size // 64), self.latent_dim),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.res_blocks(x)
        x = x.view(x.size(0), -1)
        validity = self.adv_layer(x)
        return validity
