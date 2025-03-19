import numpy as np
import tqdm
import torch

from model import NeRF, positional_encoding
from ray_sampling import get_rays, coarse_sampling, fine_sampling

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)

def volume_rendering(t, sigma, rgb) :
    '''
    t : each sampled distance (B, N_samples, 1)
    sigma : Density for each samples (B, N_samples, 1)
    rgb : Color for each samples (B, N_samples, 3)
    '''
    delta = t[:, 1:, :] - t[:, :-1, :]
    infinity = 1e10 * torch.ones_like(delta[:, 0:1, :])
    delta = torch.cat([delta, infinity], dim=1)
    delta = delta.to(device=device)
    T = torch.exp(-(torch.cumsum(delta * sigma, dim=1)))
    weight = T * (1. - torch.exp(-delta * sigma))
    C = torch.sum(weight.clone() * rgb.clone(), dim=1)
    return C, weight


def train(train_imgs, train_rays):
    '''
    train_imgs : (B,H*W,4)
    train_rays (coarse) : (B,H*W,8)
    '''
    batch_size = 1024
    learning_rate = 5*1e-4
    max_iter = 200000
    print_iter = 100
    val_iter = 500

    fn_coarse = NeRF()
    fn_coarse = fn_coarse.to(device=device, dtype=torch.float32)

    params = list(fn_coarse.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    losses = []
    loss_sum = 0

    iters = 0
    pbar = tqdm.tqdm(total=max_iter)
    while iters < max_iter :
        if iters % val_iter == 0 :
            pass

        idx = np.random.randint(len(train_imgs))
        pixel_batches = np.random.randint(0, train_imgs.shape[1], batch_size)

        # Ground-truth image
        gt = train_imgs[idx][pixel_batches]
        gt = gt.to(device=device)

        # Coarse sampling & positional encoding
        t_coarse, x_coarse = coarse_sampling(train_rays[idx][pixel_batches], stratified_sampling=True)
        N_coarse = x_coarse.shape[1]
        x_coarse = positional_encoding(x_coarse, n_freq=10, include_input=True)
        x_coarse = x_coarse.to(device=device)
        d_coarse = train_rays[idx][pixel_batches, 3:6]
        d_coarse = positional_encoding(d_coarse, n_freq=4, include_input=True)
        d_coarse = d_coarse[:, None, :]
        d_coarse = d_coarse.expand(-1, N_coarse, -1)
        d_coarse = d_coarse.to(device=device)

        # Coarse MLP
        sigma_coarse, rgb_coarse = fn_coarse(x_coarse, d_coarse)  #(B,N_coarse,1), (B,N_coarse,3)

        # Volume rendering with coarse RGB and coarse density
        coarse_color, weights = volume_rendering(t_coarse, sigma_coarse, rgb_coarse)

        # Fine sampling & positional encoding


        
        optimizer.zero_grad()
        loss = torch.mean(torch.square(coarse_color - gt[:, :3]))

        loss.backward()
        optimizer.step()
        loss_sum += loss.item()

        iters += 1
        pbar.update(1)


