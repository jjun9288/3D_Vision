import os
import numpy as np
import tqdm
import torch


from data import load_dataset
from model import NeRF, positional_encoding
from ray_sampling import get_rays, coarse_sampling, fine_sampling
from trainer import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)

current_dir = os.getcwd()
data_dir = os.path.join(current_dir, "data/nerf_synthetic/lego")
imgs, c2w, render_poses, [H, W, focal], train_test_val_split = load_dataset(data_dir)
train_imgs, test_imgs, val_imgs = imgs[train_test_val_split[0]], imgs[train_test_val_split[1]], imgs[train_test_val_split[2]]
train_f = torch.flatten(train_imgs, 1, 2)    # (B,H*W,4)
val_f = torch.flatten(val_imgs, 1, 2)
test_f = torch.flatten(test_imgs, 1, 2)
train_c2w, test_c2w, val_c2w = c2w[train_test_val_split[0]], c2w[train_test_val_split[1]], c2w[train_test_val_split[2]]

K = np.array([
    [focal, 0, 0.5*W],
    [0, focal, 0.5*H],
    [0, 0, 1]])


# All rays from all the images
train_rays = torch.from_numpy(get_rays(H, W, K, train_c2w))   # (B,H*W,8)
val_rays = torch.from_numpy(get_rays(H, W, K, val_c2w))
test_rays = torch.from_numpy(get_rays(H, W, K, test_c2w))

train(train_f, train_rays)
