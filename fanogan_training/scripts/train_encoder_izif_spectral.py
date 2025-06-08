import os
import torch
import torch.nn as nn
from torchvision.utils import save_image

import matplotlib.pyplot as plt
from PIL import Image


def train_encoder_izif(results_dir, pbar, opt, generator, discriminator, encoder, dataloader, device, kappa=1.0):
    
    # generator.load_state_dict(torch.load("results/generator"))
    # discriminator.load_state_dict(torch.load("results/discriminator"))

    generator.to(device).eval()
    discriminator.to(device).eval()
    encoder.to(device)

    criterion = nn.MSELoss()

    optimizer_E = torch.optim.Adam(encoder.parameters(), lr=opt['lr'], betas=(opt['b1'], opt['b2']))

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(f'{results_dir}/images', exist_ok=True)

    padding_epoch = len(str(opt['n_epochs']))
    padding_i = len(str(len(dataloader)))
    
    imgs_losses = []
    features_losses = []
    e_losses = []

    batches_done = 0
    for epoch in range(opt['n_epochs']):
        for i, (imgs, _, _, _) in enumerate(dataloader):

            real_imgs = imgs.to(device)
            optimizer_E.zero_grad()
            z = encoder(real_imgs)
            fake_imgs = generator(z)

            real_features = discriminator.forward_features(real_imgs)
            fake_features = discriminator.forward_features(fake_imgs)

            # izif architecture
            loss_imgs = criterion(fake_imgs, real_imgs)
            loss_features = criterion(fake_features, real_features)
            e_loss = loss_imgs + kappa * loss_features
            
            imgs_losses.append(loss_imgs.item())
            features_losses.append(loss_features.item())
            e_losses.append(e_loss.item())
            

            e_loss.backward()
            optimizer_E.step()

            
            pbar.set_description(
                f"[Epoch {epoch:{padding_epoch}}/{opt['n_epochs']}]"
                f"[Batch {i:{padding_i}}/{len(dataloader)}] "
                f"[E loss: {e_loss.item():3f}]"
            )
            pbar.update(1)
                
            if i % opt['n_critic'] == 0:
                    
                if batches_done % opt['sample_interval'] == 0:
                    # fake_z = encoder(fake_imgs)
                    # reconfiguration_imgs = generator(fake_z)
                    save_image(torch.cat([real_imgs.data[:25], fake_imgs.data[:25]], dim=0), f"{results_dir}/images/{batches_done:06}.png", nrow=5, ncols=10, normalize=True)
                    
                    fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(15, 3))
                    axes.plot(imgs_losses, label='imgs loss')
                    axes.plot(features_losses, label='features loss')
                    axes.plot(e_losses, label='e losses')
                    
                    y_min = max([max(imgs_losses[-1000:]), max(features_losses[-1000:]), max(e_losses[-1000:])])
                    y_max = min([min(imgs_losses[-1000:]), min(features_losses[-1000:]), min(e_losses[-1000:])])
                    dy = y_max - y_min
                    axes.set_ylim(y_min - 0.5 * dy, y_max + 0.5 * dy)
                    
                    axes.legend()
                    plt.savefig(f'{results_dir}/images/loss_{batches_done:06}.png')
                    

                batches_done += opt['n_critic']
                
        torch.save(encoder.state_dict(), f"{results_dir}/encoder_{epoch}.pth")
        
    torch.save(generator.state_dict(), f'{results_dir}/encoder.pth')
