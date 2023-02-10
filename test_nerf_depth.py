import os
import os.path as osp
from tqdm import tqdm
import yaml
import argparse
import numpy as np
import imageio
import open3d as o3d

import torch
import torch.nn as nn

from model import FourierEmbedding, NeRF
from camera import Camera, Rays, convert_rays_to_ndc, restore_ndc_points
from render import nerf_render
from utils.visual_util import build_colored_pointcloud


# Create tensor on GPU by default ('.to(device)' & '.cuda()' cost time!)
torch.set_default_tensor_type('torch.cuda.FloatTensor')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Config files')
    parser.add_argument('--checkpoint', type=int, default=200000, help='Checkpoint (iteration) to load')

    # Read parameters
    args = parser.parse_args()
    with open(args.config) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    for ckey, cvalue in configs.items():
        args.__dict__[ckey] = cvalue

    # Fix the random seed
    seed = args.random_seed
    np.random.seed(seed)
    torch.manual_seed(seed)


    # Load the data
    if args.dataset_type == 'blender':
        from data_loader.load_blender import load_blender_data
        imgs_test, _, poses_test, img_h, img_w, focal = load_blender_data(data_root=args.data_root,
                                                                          split='test',
                                                                          half_res=args.half_res,
                                                                          white_bkgd=args.white_bkgd)
        n_view_test = imgs_test.shape[0]

        near = 2.
        far = 6.

    elif args.dataset_type == 'llff':
        from data_loader.load_llff import load_llff_data

        imgs, poses, bounds, img_h, img_w, focal = load_llff_data(data_root=args.data_root,
                                                                  factor=args.factor,
                                                                  recenter=True,
                                                                  bound_factor=0.75)

        # Split train & val (test)
        test_ids = np.arange(imgs.shape[0])[::args.hold_for_test]
        imgs_test, poses_test = imgs[test_ids], poses[test_ids]
        n_view_test = imgs_test.shape[0]

        # Define bounds
        if args.use_ndc:
            near = 0.
            far = 1.
        else:
            near = bounds.min() * 0.9
            far = bounds.max()

    else:
        raise ValueError('Not implemented!')

    # Create the Fourier embedding
    point_embedding = FourierEmbedding(n_freq=args.n_freq_point)
    view_embedding = FourierEmbedding(n_freq=args.n_freq_view)

    # Create the network (coarse) and load trained model weights
    model = NeRF(n_layer=args.n_layer,
                 n_dim=args.n_dim,
                 input_dim=point_embedding.output_dim,
                 input_view_dim=view_embedding.output_dim,
                 skips=[4],
                 use_viewdir=args.use_viewdir,
                 rgb_act=args.rgb_act,
                 density_act=args.density_act)
    weight_path = osp.join(args.exp_base, 'model_%06d.pth.tar'%(args.checkpoint))
    # weight_path = osp.join(args.exp_base, 'model_%06d_fine.pth.tar'%(args.checkpoint))
    model.load_state_dict(torch.load(weight_path))

    # Create the network (fine) and load trained model weights
    model_fine = None
    if args.n_sample_point_fine > 0:
        model_fine = NeRF(n_layer=args.n_layer,
                          n_dim=args.n_dim,
                          input_dim=point_embedding.output_dim,
                          input_view_dim=view_embedding.output_dim,
                          skips=[4],
                          use_viewdir=args.use_viewdir,
                          rgb_act=args.rgb_act,
                          density_act=args.density_act)
        weight_path_fine = osp.join(args.exp_base, 'model_%06d_fine.pth.tar'%(args.checkpoint))
        model_fine.load_state_dict(torch.load(weight_path_fine))

    # Create the loss
    img_loss = nn.MSELoss(reduction='mean')

    # Create path to save rendered images
    exp_base = args.exp_base
    save_render_base = osp.join(exp_base, 'test_%06d'%(args.checkpoint))
    os.makedirs(save_render_base, exist_ok=True)


    """
    Traverse the testing set
    """
    tbar = tqdm(total=n_view_test)
    for vid in range(n_view_test):
        target = torch.Tensor(imgs_test[vid])
        pose = torch.Tensor(poses_test[vid])

        # Get rays for all pixels
        cam = Camera(img_h, img_w, focal, pose)
        rays_o, rays_d = cam.get_rays()
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
        viewdirs = rays_d / rays_d.norm(dim=1, keepdim=True)
        if args.use_ndc:
            rays_o, rays_d = convert_rays_to_ndc(rays_o, rays_d, img_h, img_w, focal, near_plane=1.)

        # Batchify
        rgb_map, rgb_map_fine = [], []
        depth_map, depth_map_fine = [], []
        acc_map, acc_map_fine = [], []
        for i in range(0, rays_o.shape[0], args.chunk):
            # Forward
            with torch.no_grad():
                rays_o_batch = rays_o[i:(i + args.chunk)]
                rays_d_batch = rays_d[i:(i + args.chunk)]
                viewdirs_batch = viewdirs[i:(i + args.chunk)]
                rays = Rays(rays_o_batch, rays_d_batch, viewdirs_batch,
                            args.n_sample_point, args.n_sample_point_fine, near, far, args.perturb)
                ret_dict = nerf_render(rays, point_embedding, view_embedding, model, model_fine,
                                       density_noise_std=0.,
                                       white_bkgd=args.white_bkgd)

                rgb_map.append(ret_dict['rgb_map'])
                depth_map.append(ret_dict['depth_map'])
                acc_map.append(ret_dict['acc_map'])
                rgb_map_fine.append(ret_dict['rgb_map_fine'])
                depth_map_fine.append(ret_dict['depth_map_fine'])
                acc_map_fine.append(ret_dict['acc_map_fine'])

        rgb_map = torch.cat(rgb_map, 0)
        depth_map = torch.cat(depth_map, 0)
        acc_map = torch.cat(acc_map, 0)
        rgb_map_fine = torch.cat(rgb_map_fine, 0)
        depth_map_fine = torch.cat(depth_map_fine, 0)
        acc_map_fine = torch.cat(acc_map_fine, 0)

        # Cast depth to 3D points
        points = rays_o + depth_map.unsqueeze(1) * rays_d
        points_fine = rays_o + depth_map_fine.unsqueeze(1) * rays_d
        if args.use_ndc:
            points = restore_ndc_points(points, img_h, img_w, focal, near_plane=1.)
            points_fine = restore_ndc_points(points_fine, img_h, img_w, focal, near_plane=1.)

        points = points.cpu().numpy()
        rgb_map = rgb_map.cpu().numpy().clip(0., 1.)
        acc_map = acc_map.cpu().numpy()
        points_fine = points_fine.cpu().numpy()
        rgb_map_fine = rgb_map_fine.cpu().numpy().clip(0., 1.)
        acc_map_fine = acc_map_fine.cpu().numpy()

        # Filter out empty points
        acc_thresh = 0.99
        valid = (acc_map > acc_thresh)
        points, rgb_map = points[valid], rgb_map[valid]
        valid_fine = (acc_map_fine > acc_thresh)
        points_fine, rgb_map_fine = points_fine[valid_fine], rgb_map_fine[valid_fine]

        # Visualize
        cam_pose = pose.cpu().numpy()
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        coord_frame.transform(cam_pose)

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])

        # pcds = [build_colored_pointcloud(points, rgb_map)]
        # pcds.append(coord_frame)
        # o3d.visualization.draw_geometries(pcds)
        # pcds = [build_colored_pointcloud(points_fine, rgb_map_fine)]
        # pcds.append(coord_frame)
        # o3d.visualization.draw_geometries(pcds)

        # vis.add_geometry(build_colored_pointcloud(points, rgb_map))
        vis.add_geometry(build_colored_pointcloud(points_fine, rgb_map_fine))
        vis.add_geometry(coord_frame)
        vis.run()
        vis.destroy_window()

        tbar.update(1)