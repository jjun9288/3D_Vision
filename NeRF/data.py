import os
import imageio.v2 as imageio
import json
import numpy as np
import torch

current_dir = os.getcwd()

# Blender dataset
data_dir = os.path.join(current_dir, 'NeRF/data/nerf_synthetic/lego')


'''
Camera coordinate to World coordinate
'''
# Translate along z-axis
trans_t = lambda t : torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]]).float()

# Rotate around y-axis
rot_phi = lambda phi : torch.Tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]]).float()

# Rotate around x-axis
rot_theta = lambda theta : torch.Tensor([
    [np.cos(theta), 0, -np.sin(theta), 0],
    [0, 1, 0, 0],
    [np.sin(theta), 0, np.cos(theta), 0],
    [0, 0, 0, 1]]).float()

'''
Poses to render
'''
def pose_spherical(theta, phi, radius) :
    # Representing the 3D coorindate with spherical coordinate
    translate = trans_t(radius)
    rotate_y = rot_phi(phi / (180.*np.pi)) @ translate
    rotate_x = rot_theta(theta / (180.*np.pi)) @ rotate_y
    c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ rotate_x    # Flip x-axis and switch y-axis / z-axis
    return c2w


def load_dataset() :
    splits = ['train', 'test', 'val']
    
    metas = {}
    for split in splits : 
        with open(os.path.join(data_dir, 'transforms_{}.json'.format(split)), 'r') as fp :
            metas[split] = json.load(fp)
    
    all_imgs = []
    all_poses = []
    cnts = [0]
    for split in splits : 
        meta = metas[split]
        imgs = []
        poses = []

        for frame in meta['frames']:
            img_dir = os.path.join(data_dir, frame['file_path'] + '.png')
            img = imageio.imread(img_dir)
            imgs.append(img)
            
            pose = np.array(frame['transform_matrix'])
            poses.append(pose)
        
        imgs = (np.array(imgs) / 255.).astype(np.float32)   # (Batch, H, W, 4)
        poses = np.array(poses).astype(np.float32)
        cnts.append(cnts[-1] + imgs.shape[0])

        all_imgs.append(imgs)
        all_poses.append(poses)

    train_test_val_split = [np.arange(cnts[i], cnts[i+1]) for i in range(3)]

    all_imgs = np.concatenate(all_imgs, axis=0)
    all_poses = np.concatenate(all_poses, axis=0)

    ''''
    Computing focal length
    (Width/2) / focal = tan (theta/2)  -> focal = Width / (2 * tan(theta/2))'
    '''
    camera_angle = float(meta['camera_angle_x'])
    H, W = all_imgs[0].shape[:2]
    focal = W / (2. * np.tan(camera_angle / 2.))
    
    thetas = np.linspace(-180, 180, 41)[:-1]    # 41개로 나누면 이쁜 숫자로 딱 떨어진다! 그리고 시작(-180)과 끝(180)은 같으므로 하나는 뺀다.
    render_poses = torch.stack([pose_spherical(theta, -30.0, 4.0) for theta in thetas], dim=0)

    return all_imgs, all_poses, render_poses, [H, W, focal], train_test_val_split