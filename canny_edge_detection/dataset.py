import os
import sys

import pandas as pd
import numpy as np

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

import torchvision

from tqdm.notebook import tqdm

# google tiles data
# GT_DIR = '/home/jupyter/datasphere/datasets/googlemaps-dataset-140k/googlemaps_data'

# GT_NEGATIVE_PATHS = (
#     sorted([
#         f'{GT_DIR}/negative_v2/negative_samples_v2/{s}' 
#         for s in os.listdir(f'{GT_DIR}/negative_v2/negative_samples_v2') 
#         if s.endswith('.png')
#     ]) 
#     + sorted([
#         f'{GT_DIR}/negative_v3/negative_samples_v3/{s}' 
#         for s in os.listdir(f'{GT_DIR}/negative_v3/negative_samples_v3') 
#         if s.endswith('.png')
#     ])
# )
# GT_POSITIVE_PATHS = [
#     f'{GT_DIR}/positive_samples/{s}' 
#     for s in os.listdir(f'{GT_DIR}/positive_samples') 
#     if s.endswith('.png')
# ]

# gt_negative_paths_set = set(GT_NEGATIVE_PATHS)

# GT_NEGATIVE_COORDS = (
#     [
#         (lat, lon) 
#         for _, (iid, lat, lon) in pd.read_csv(f'{GT_DIR}/negative_v2/negative_coords_v2.csv').iterrows()
#         if f'{GT_DIR}/negative_v2/negative_samples_v2/{str(int(iid)).zfill(6)}.png' in gt_negative_paths_set
#     ]
#     + [
#         (lat, lon) 
#         for _, (iid, lat, lon) in pd.read_csv(f'{GT_DIR}/negative_v3/negative_coords_v3.csv').iterrows()
#         if f'{GT_DIR}/negative_v3/negative_samples_v3/{str(int(iid)).zfill(6)}.png' in gt_negative_paths_set
#     ]
# )

# positive_coords_dct = {
#      f'{GT_DIR}/positive_samples/pos_{iid}.png' : (lat, lon)
#     for _, (iid, _, _, lon, lat, _, _, _, _, _, _, _) 
#     in pd.read_csv(f'{GT_DIR}/positive_samples_coords.csv').iterrows()
# }

# GT_POSITIVE_COORDS = [positive_coords_dct[path] for path in GT_POSITIVE_PATHS]

# google satellite data
GS_DIR = '/home/jupyter/datasphere/datasets/sasgis-dataset-100k/google_satellite_data'

GS_NEGATIVE_PATHS = sorted([
    f'{GS_DIR}/negative_samples/{s}'
    for s in os.listdir(f'{GS_DIR}/negative_samples') 
    if s.endswith('.jpeg')
]) 
GS_POSITIVE_PATHS = sorted([
    f'{GS_DIR}/positive_samples/{s}'
    for s in os.listdir(f'{GS_DIR}/positive_samples') 
    if s.endswith('.jpeg')
])

GS_NEGATIVE_COORDS = [
    (lat, lon) 
    for _, (lat, lon, rect) in pd.read_csv(f'{GS_DIR}/negative_coords_new.csv').iterrows()
]
GS_POSITIVE_COORDS = [
    (lat, lon) 
    for _, (iid, _, _, lon, lat, _, _, _, _, _, _, _) 
    in pd.read_csv(f'{GS_DIR}/Eastern_desert_archaeological_structures_NEW.csv').iterrows()
]

class ArchDataset(Dataset):
    
    def __init__(self, img_paths, coords=None, transform=None, anomalies=False):
        self.img_paths = img_paths
        self.coords = coords
        self.transform = transform
        self.anomalies = anomalies

    def __len__(self):
        return len(self.img_paths)
    
    def get_coords(self, idx):
        if self.coords:
            return self.coords[idx]
        else:
            raise NotImplemented()

    def __getitem__(self, idx):
        img_name = self.img_paths[idx]
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image, idx, self.coords[idx], int(self.anomalies)
