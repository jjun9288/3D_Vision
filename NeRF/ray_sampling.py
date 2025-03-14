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

    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
    i, j = i.t(), j.t()

    i2c = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], dim=-1)    #image coordinate to camera coordinate, (H,W,3)
    # Rotate ray directions from camera coordinate to world coordinate
    rays_d = torch.sum(i2c[..., np.newaxis, :] * c2w[:3, :3], dim=-1)   # (H,W,1,3)
    rays_d = rays_d.reshape((-1,3)).float()                             # (H*W,3)
    # Translate camera coordinate's origin to world coordinate  
    rays_o = c2w[:3, -1].expand(rays_d.shape)                           # (H*W,3)

    near, far = 0., 1.
    near *= torch.ones_like(rays_d[...,:1])  # (H*W,1)
    far *= torch.ones_like(rays_d[...,:1])   # (H*W,1)

    rays = torch.cat([rays_o, rays_d, near, far], dim=-1)   # (H*W,8)
    
    return rays

'''
def batchify_rays(rays, chunk=1024*32) :
'''



def sample_rays(ray, N_samples, stratified_sampling=False) :
    '''
    Fine sampling the rays
    ray : (H*W, 8)
    N_samples : Number of points to sample along each rays
    stratified_sampling : Random sampling applied on each intervals of rays
    '''

    N_rays = ray.shape[0]
    rays_o, rays_d, near, far = ray[:,0:3], ray[:,3:6], ray[:,6], ray[:,7]

    # Ray steps
    t_vals = torch.linspace(0., 1., steps=N_samples)
    z_vals = (near * (1-t_vals)) + (far * t_vals)
    z_vals = z_vals.expand((N_rays, N_samples))

    # Stratified sampling
    if stratified_sampling :
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat(mids, z_vals[...,-1:])    # keep dimensions
        lower = torch.cat(mids, z_vals[...,:1])
        t_rand = torch.rand(z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand
    
    # x = o + td
    samples = rays_o[..., None, :] + (z_vals[..., :, None] * rays_d[..., None, :])  #[N_rays, N_samples, 3]

    return samples