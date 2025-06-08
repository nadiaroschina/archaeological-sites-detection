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

import lpips

from torchvision.models import vit_b_16, ViT_B_16_Weights

from sklearn.metrics.pairwise import cosine_similarity

vit_model = vit_b_16(weights='ViT_B_16_Weights.IMAGENET1K_V1')
vit_model.heads = torch.nn.Identity()
vit_model.eval();

import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from dataset import negatives_dataset, positives_dataset, VANILLA_TRANSFORM_128

transform = VANILLA_TRANSFORM_128

loss_fn_alex = lpips.LPIPS(net='alex')
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_image(names):
    imgs = []
    for name in names:
        img = transform(Image.open(name).convert('RGB'))
        imgs.append(img)
    return imgs

def align_image(img):
    return (img - img.min()) / (img.max() - img.min())

def draw_image(imgs):
    fig, axes = plt.subplots(ncols=len(imgs) , figsize=(len(imgs) * 2, 2))

    if len(imgs) == 1:
        img = imgs[0]
        axes.imshow(align_image(img).permute(1, 2, 0))
    else:
        for i, img in enumerate(imgs): 
            axes[i].imshow(align_image(img).permute(1, 2, 0))
    plt.show()
    
    
def vit_cosine_similarity(img1, img2):

    assert len(img1.shape) == len(img2.shape) == 3, f'{img1.shape=} {img2.shape=}'

    # print(f'input: {img1.shape=} {img2.shape=}')

    resize = transforms.Resize(size=224)

    img1 = resize(img1.unsqueeze(0))
    img2 = resize(img2.unsqueeze(0))

    # print(f'resized: {img1.shape=} {img2.shape=}')

    features1 = vit_model(img1).detach().numpy()
    features2 = vit_model(img2).detach().numpy()

    # print(f'features {features1.shape=} {features2.shape=}')

    sim = cosine_similarity(features1, features2)
    return sim[0, 0].item()



@torch.no_grad()
def apply_model(model, n_neg=2, n_pos=2, other_image_names=['cat.png']):
    
    negative_examples = torch.stack([negatives_dataset[i].to(device) for i in np.random.randint(0, 100, n_neg)], dim=0)
    positive_examples = torch.stack([positives_dataset[i].to(device) for i in np.random.randint(0, 100, n_pos)], dim=0)
    other_examples =  torch.stack([anomaly.to(device) for anomaly in load_image(other_image_names)], dim=0)

    examples = torch.concat((negative_examples, positive_examples, other_examples), dim=0)
    
    print(f'{len(examples)=}')
    
    reconstructions = model(examples)[0]

    titles = ['neg' for _ in range(n_neg)] + ['pos' for _ in range(n_pos)] + ['extra' for _ in range(len(other_image_names))]
    return examples, reconstructions, titles

@torch.no_grad()
def anomaly_detection(examples, reconstructions, titles):
    ncols = len(examples)
    nrows = 2
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(15, 4))

    data = torch.cat([examples[None], reconstructions[None]], dim=0)
    print(data.shape)
    for row in range(nrows):
        for col in range(ncols):
            if row == 1:
                
                axes[row - 1, col].set_title(f'{titles[col]}')

                similarity_cosine = vit_cosine_similarity(data[row - 1, col].to('cpu'), data[row, col].to('cpu'))
                similarity_lpips = loss_fn_alex(data[row - 1, col].to('cpu'), data[row, col].to('cpu'))[0, 0, 0, 0].item()
                axes[row, col].set_title(f'lpips: {np.round(similarity_lpips, 2)} cosine: {np.round(similarity_cosine, 2)}')
            
            axes[row, col].imshow(align_image(data[row, col].to('cpu')).permute(1, 2, 0))
            axes[row, col].axis('off')
    plt.show()