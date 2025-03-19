import torch
import numpy as np

def get_rays(H, W, K, c2w) :
    '''
    H, W : Image height & width
    K : Camera's intrinsic matrix
        [[f, 0, cx],
         [0, f, cy],
         [0, 0,  1]]
    c2w : Transformation matrix from camera coordinate to world coordinate
    '''

    B = c2w.shape[0]    # Batch size

    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    i, j = i.transpose(), j.transpose()

    i2c = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], axis=-1)    #image coordinate to camera coordinate, (H,W,3)
    # Rotate ray directions from camera coordinate to world coordinate
    R = c2w[..., :3, :3]
    R = R[:, None, None, :, :]
    rays_d = np.sum(i2c[..., np.newaxis, :] * R, axis=-2)   # (B,H,W,1,3)
    rays_d = rays_d.reshape((B,-1,3))                             # (B*H*W,3)
    # Translate camera coordinate's origin to world coordinate  
    #rays_o = c2w[:3, -1].expand(rays_d.shape)  
    rays_o = c2w[..., :3, -1]           
    rays_o = np.broadcast_to(rays_o[:, None,  ], np.shape(rays_d))             # (B*H*W,3)
    
    near, far = 0., 1.
    near *= np.ones_like(rays_d[...,:1])  # (B*H*W,1)
    far *= np.ones_like(rays_d[...,:1])   # (B*H*W,1)

    rays = np.concatenate([rays_o, rays_d, near, far], axis=-1)   # (B*H*W,8)
    
    return rays


def coarse_sampling(ray, N_samples=64, stratified_sampling=False) :
    '''
    Fine sampling the rays
    ray : (H*W, 8)
    N_samples : Number of points to sample along each rays
    stratified_sampling : Random sampling applied on each intervals of rays
    '''

    N_rays = ray.shape[0]
    rays_o, rays_d, near, far = ray[:,0:3], ray[:,3:6], ray[:,6], ray[:,7]
    near, far = near.reshape((-1,1)), far.reshape((-1,1))

    # Ray steps
    t_vals = torch.linspace(0., 1., steps=N_samples)
    z_vals = (near * (1-t_vals)) + (far * t_vals)
    #z_vals = z_vals.expand((N_rays, N_samples))

    # Stratified sampling
    if stratified_sampling :
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat((mids, z_vals[...,-1:]), dim=-1)    # keep dimensions
        lower = torch.cat((mids, z_vals[...,:1]), dim=-1)
        t_rand = torch.rand(z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand
    
    # x = o + td
    samples = rays_o[..., None, :] + (z_vals[..., :, None] * rays_d[..., None, :])  #[N_rays, N_samples, 3]
    z_vals = z_vals[:, :, None]
    return z_vals, samples


def fine_sampling () :
    pass