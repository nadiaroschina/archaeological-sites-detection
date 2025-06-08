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


def train_epoch(pbar, num_epoch, loss_function, checkpoint_path, optimizer_path, samples_dir):
    
    model.train()
    train_rec, train_reg, train_loss = 0, 0, 0
    
    for batch_idx, data in enumerate(loader):
        
        data = data.to(device)
        optimizer.zero_grad()
        
        recon_batch, mu, logvar = model(data)
        rec_loss, reg_loss = loss_function(recon_batch, data, mu, logvar)
        loss = rec_loss + reg_loss
        
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())
        
        wandb.log({'loss': loss.item()})
        
        train_loss += loss.item()
        train_rec += rec_loss.item()
        train_reg += reg_loss.item()
    
        pbar.set_description(f"[REC: {rec_loss.item()/len(data) :.2f}] [REG: {reg_loss.item()/len(data): .2f}]")
        pbar.update(1)

        if batch_idx % 50 == 0:
            
            torch.save(model.state_dict(), checkpoint_path)
            torch.save(optimizer.state_dict(), optimizer_path)
            
            with torch.no_grad():
                sample_img_file = f'{samples_dir}/sample_e{str(num_epoch).zfill(2)}_{str(batch_idx).zfill(3)}.png'
                utils.save_image(
                    model.sample(16).cpu().data,
                    sample_img_file,
                    normalize=True,
                    nrow=4,
                )
                wandb.log({'sample': wandb.Image(Image.open(sample_img_file))})
            