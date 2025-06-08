import os
import sys

import pandas as pd
import numpy as np

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms, utils
from torchvision.utils import make_grid

import matplotlib.pyplot as plt
from tqdm.auto import tqdm


VANILLA_TRANSFORM_128 = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

VANILLA_TRANSFORM_64 = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


PATH_NEGATIVE_V2 = '/home/jupyter/datasphere/datasets/googlemaps-dataset-140k/googlemaps_data/negative_v2'
PATH_NEGATIVE_V3 = '/home/jupyter/datasphere/datasets/googlemaps-dataset-140k/googlemaps_data/negative_v3'
PATH_POSITIVE = '/home/jupyter/datasphere/datasets/googlemaps-dataset-140k/googlemaps_data/positive_samples'

IMG_FILES_V2 = [f'{PATH_NEGATIVE_V2}/negative_samples_v2/{filename}' for filename in os.listdir(f'{PATH_NEGATIVE_V2}/negative_samples_v2') if '.png' in filename]
IMG_FILES_V3 = [f'{PATH_NEGATIVE_V3}/negative_samples_v3/{filename}' for filename in os.listdir(f'{PATH_NEGATIVE_V3}/negative_samples_v3')if '.png' in filename]

IMG_FILES = IMG_FILES_V2 + IMG_FILES_V3


class ArchNegatives(Dataset):
    
    def __init__(self, transform=None, filenames=IMG_FILES, coords=None):
        self.transform = transform
        self.filenames = filenames
        self.coords = coords

    def __len__(self):
        return len(self.filenames)
    
    def get_coords(self, idx):
        if self.coords:
            return self.coords[idx]
        else:
            raise NotImplemented()

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)

        return image
    

class ArchPositives(Dataset):
    def __init__(self, transform=None):
        self.root_dir = PATH_POSITIVE
        self.transform = transform
        self.images = [s for s in os.listdir(self.root_dir) if s.endswith('png')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)

        return image
    


negatives_dataset = ArchNegatives(transform=VANILLA_TRANSFORM_128)
negatives_loader = DataLoader(negatives_dataset, batch_size=256, shuffle=True)

positives_dataset = ArchPositives(transform=VANILLA_TRANSFORM_128)
positives_loader = DataLoader(positives_dataset, batch_size=256, shuffle=False)