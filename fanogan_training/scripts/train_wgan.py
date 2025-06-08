import os
import torch
import torch.autograd as autograd
from torchvision.utils import save_image

import wandb
from PIL import Image


def compute_gradient_penalty(D, real_samples, fake_samples, device):
    '''Calculates the gradient penalty loss for WGAN GP'''
    # random weight term for interpolation between real and fake samples
    alpha = torch.rand(*real_samples.shape[:2], 1, 1, device=device)
    # random interpolation between real and fake samples
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples)
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    d_interpolates = D(interpolates)
    fake = torch.ones(*d_interpolates.shape, device=device)
    gradients = autograd.grad(
        outputs=d_interpolates, inputs=interpolates, grad_outputs=fake, create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    gradients = gradients.view(gradients.shape[0], -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def train_wgangp(results_dir, pbar, opt, generator, discriminator, dataloader, device, lambda_gp=10):
    
    generator.to(device)
    discriminator.to(device)

    optimizer_G = torch.optim.Adam(
        generator.parameters(), 
        lr=opt['lr'], betas=(opt['b1'], opt['b2']), weight_decay=opt['weight_decay']
    )
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(), 
        lr=opt['lr'], betas=(opt['b1'], opt['b2']), weight_decay=opt['weight_decay']
    )

    os.makedirs(f'{results_dir}/images', exist_ok=True)
    os.makedirs(f'{results_dir}/used_indices', exist_ok=True)

    padding_epoch = len(str(opt['n_epochs']))
    padding_i = len(str(len(dataloader)))
    

    batches_done = 0
    for epoch in range(opt['n_epochs']):
        
        indices_used = []
        
        for i, (imgs, indices, coords, labels) in enumerate(dataloader):

            real_imgs = imgs.to(device)
            indices_used.append((i, indices))
            
            optimizer_D.zero_grad()
            z = torch.randn(imgs.shape[0], opt['latent_dim'], device=device)
            fake_imgs = generator(z)

            real_validity = discriminator(real_imgs)
            fake_validity = discriminator(fake_imgs.detach())
            gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data, device)
            d_loss = (-torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty)

            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()

            # Train the generator and output log every n_critic steps
            if i % opt['n_critic'] == 0:

                fake_imgs = generator(z)
                fake_validity = discriminator(fake_imgs)
                g_loss = -torch.mean(fake_validity)

                g_loss.backward()
                optimizer_G.step()

                pbar.set_description(f"[Epoch {epoch:{padding_epoch}}/{opt['n_epochs']}]"
                      f"[Batch {i:{padding_i}}/{len(dataloader)}] "
                      f"[D loss: {d_loss.item():3f}] "
                      f"[G loss: {g_loss.item():3f}]")
                pbar.update(1)
                
                wandb.log({'d_loss': d_loss.item()})
                wandb.log({'g_loss': g_loss.item()})
                wandb.log({'loss': d_loss.item() + g_loss.item()})

                if batches_done % opt['sample_interval'] == 0:
                    save_image(fake_imgs.data[:25], f'{results_dir}/images/{batches_done:06}.png', nrow=5, normalize=True)
                    wandb.log({'sample': wandb.Image(Image.open(f'{results_dir}/images/{batches_done:06}.png'))})

                batches_done += opt['n_critic']
        
        torch.save(generator.state_dict(), f'{results_dir}/generator_{epoch}.pth')
        torch.save(discriminator.state_dict(), f'{results_dir}/discriminator_{epoch}.pth')
        
        with open(f'{results_dir}/used_indices/epoch_{epoch}.csv', 'w') as f:
            s = ','.join(map(str, ['step'] + [f'index_{j}' for j in range(len(indices))])) + '\n'
            f.write(s)
            for (i, indices) in indices_used:
                s = ','.join(map(str, [i] + [int(ind) for ind in indices])) + '\n'
                f.write(s)
            
    torch.save(generator.state_dict(), f'{results_dir}/generator.pth')
    torch.save(discriminator.state_dict(), f'{results_dir}/discriminator.pth')
