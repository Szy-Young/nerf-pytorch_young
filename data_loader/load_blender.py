import os
import os.path as osp
import torch
import numpy as np
import imageio 
import json
import cv2


# trans_t = lambda t : torch.Tensor([
#     [1,0,0,0],
#     [0,1,0,0],
#     [0,0,1,t],
#     [0,0,0,1]]).float()
#
# rot_phi = lambda phi : torch.Tensor([
#     [1,0,0,0],
#     [0,np.cos(phi),-np.sin(phi),0],
#     [0,np.sin(phi), np.cos(phi),0],
#     [0,0,0,1]]).float()
#
# rot_theta = lambda th : torch.Tensor([
#     [np.cos(th),0,-np.sin(th),0],
#     [0,1,0,0],
#     [np.sin(th),0, np.cos(th),0],
#     [0,0,0,1]]).float()
#
#
# def pose_spherical(theta, phi, radius):
#     c2w = trans_t(radius)
#     c2w = rot_phi(phi/180.*np.pi) @ c2w
#     c2w = rot_theta(theta/180.*np.pi) @ c2w
#     c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
#     return c2w
#
#
# def load_blender_data_old(basedir, half_res=False, testskip=1):
#     splits = ['train', 'val', 'test']
#     metas = {}
#     for s in splits:
#         with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
#             metas[s] = json.load(fp)
#
#     all_imgs = []
#     all_poses = []
#     counts = [0]
#     for s in splits:
#         meta = metas[s]
#         imgs = []
#         poses = []
#         if s=='train' or testskip==0:
#             skip = 1
#         else:
#             skip = testskip
#
#         for frame in meta['frames'][::skip]:
#             fname = os.path.join(basedir, frame['file_path'] + '.png')
#             imgs.append(imageio.imread(fname))
#             poses.append(np.array(frame['transform_matrix']))
#         imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
#         poses = np.array(poses).astype(np.float32)
#         counts.append(counts[-1] + imgs.shape[0])
#         all_imgs.append(imgs)
#         all_poses.append(poses)
#
#     i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
#
#     imgs = np.concatenate(all_imgs, 0)
#     poses = np.concatenate(all_poses, 0)
#
#     img_h, img_w = imgs[0].shape[:2]
#     camera_angle_x = float(meta['camera_angle_x'])
#     focal = .5 * img_w / np.tan(.5 * camera_angle_x)
#
#     render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
#
#     if half_res:
#         img_h = img_h//2
#         img_w = img_w//2
#         focal = focal/2.
#
#         imgs_half_res = np.zeros((imgs.shape[0], img_h, img_w, 4))
#         for i, img in enumerate(imgs):
#             imgs_half_res[i] = cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_AREA)
#         imgs = imgs_half_res
#
#     return imgs, poses, render_poses, [img_h, img_w, focal], i_split


def load_blender_data(data_root,
                      split='test',
                      half_res=False,
                      white_bkgd=False,
                      load_depth=False,
                      testskip=1,
                      near=2.,
                      far=6.):
    # Load meta info
    with open(osp.join(data_root, 'transforms_%s.json'%(split)), 'r') as f:
        meta = json.load(f)

    # Skip some views during validation/testing
    if split == 'train' or testskip == 0:
        skip = 1
    else:
        skip = testskip

    imgs, poses = [], []
    for frame in meta['frames'][::skip]:
        img_file = osp.join(data_root, frame['file_path'] + '.png')
        img = imageio.imread(img_file)
        imgs.append(img)
        pose = np.array(frame['transform_matrix'])
        poses.append(pose)
    imgs = np.stack(imgs, 0)        # (N, H, W, 4)
    imgs = (imgs / 255.).astype(np.float32)
    poses = np.stack(poses, 0).astype(np.float32)      # (N, 4, 4)

    if load_depth:
        depths = []
        for frame in meta['frames'][::skip]:
            depth_file = osp.join(data_root, frame['file_path'] + '_depth_0001.png')
            depth = cv2.imread(depth_file, 0)
            depths.append(depth)
        depths = np.stack(depths, 0)        # (N, H, W)
        depths = near + (far - near) * (1. - (depths / 255.).astype(np.float32))
    else:
        depths = None

    img_h, img_w = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = 0.5 * img_w / np.tan(0.5 * camera_angle_x)
    
    if half_res:
        img_h = img_h // 2
        img_w = img_w // 2
        focal = focal / 2.

        imgs_half_res = np.zeros((imgs.shape[0], img_h, img_w, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res

        if depths is not None:
            depths_half_res = np.zeros((depths.shape[0], img_h, img_w))
            for i, depth in enumerate(depths):
                depths_half_res[i] = cv2.resize(depth, (img_w, img_h), interpolation=cv2.INTER_AREA)
            depths = depths_half_res

    if white_bkgd:
        imgs = imgs[..., :3] * imgs[..., 3:] + (1. - imgs[..., 3:])
    else:
        imgs = imgs[..., :3]

    return imgs, depths, poses, img_h, img_w, focal


if __name__ == '__main__':
    data_root = '/home/ziyang/Desktop/Datasets/nerf_dataset/nerf_synthetic/lego'
    split = 'test'
    half_res = False    # True
    load_depth = True
    testskip = 20
    imgs, depths, poses, img_h, img_w, focal = load_blender_data(data_root=data_root,
                                                                 split=split,
                                                                 half_res=half_res,
                                                                 load_depth=load_depth,
                                                                 testskip=testskip)
    imgs = torch.Tensor(imgs)
    depths = torch.Tensor(depths)
    poses = torch.Tensor(poses)

    from camera import Camera
    cameras = [Camera(img_h, img_w, focal, pose) for pose in poses]

    # Accumulate point clouds from multi-view RGBD to check consistency
    n_view = len(cameras)
    pcs, colors = [], []
    for v in range(n_view):
        img, depth = imgs[v], depths[v]
        camera = cameras[v]

        u, v = torch.meshgrid(torch.arange(img_w), torch.arange(img_h))
        u, v = u.t(), v.t()

        # Only collect points from non-transparent regions
        valid = (img[:, :, 3] > 0)
        u, v = u[valid], v[valid]
        depth = depth[valid]
        pc = camera.back_project(u, v, depth)
        pcs.append(pc)
        color = img[valid][:, :3]
        colors.append(color)

    import open3d as o3d
    from visual_util import build_colored_pointcloud
    # Aggregate
    pc = torch.cat(pcs, 0).numpy()
    color = torch.cat(colors, 0).numpy()
    print(pc.shape)
    pcds = [build_colored_pointcloud(pc, color)]
    o3d.visualization.draw_geometries(pcds)