import os
import torch
import torch.nn as nn
from torchvision.utils import save_image

import wandb
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

    batches_done = 0
    for epoch in range(opt['n_epochs']):
        for i, (imgs, _) in enumerate(dataloader):

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

            e_loss.backward()
            optimizer_E.step()

            
            pbar.set_description(
                f"[Epoch {epoch:{padding_epoch}}/{opt['n_epochs']}]"
                f"[Batch {i:{padding_i}}/{len(dataloader)}] "
                f"[E loss: {e_loss.item():3f}]"
            )
            pbar.update(1)

            wandb.log({'e_loss': e_loss.item()})
                
            if i % opt['n_critic'] == 0:
                    
                if batches_done % opt['sample_interval'] == 0:
                    fake_z = encoder(fake_imgs)
                    reconfiguration_imgs = generator(fake_z)
                    save_image(reconfiguration_imgs.data[:25], f"{results_dir}/images/{batches_done:06}.png", nrow=5, normalize=True)
                    wandb.log({'sample': wandb.Image(Image.open(f"{results_dir}/images/{batches_done:06}.png"))})

                batches_done += opt['n_critic']
                
        torch.save(encoder.state_dict(), f"{results_dir}/encoder_{epoch}.pth")
        
    torch.save(generator.state_dict(), f'{results_dir}/encoder.pth')
