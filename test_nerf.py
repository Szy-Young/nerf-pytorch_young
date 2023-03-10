import os
import os.path as osp
from tqdm import tqdm
import yaml
import argparse
import numpy as np
import imageio

import torch
import torch.nn as nn

from model import FourierEmbedding, NeRF
from camera import Camera, Rays, convert_rays_to_ndc
from render import nerf_render
from metric import mse_to_psnr
from utils.pytorch_util import AverageMeter


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
    fine_sampling = (args.n_sample_point_fine > 0)
    model_fine = None
    if fine_sampling and args.two_model_for_fine:
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
    eval_meter = AverageMeter()
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
        for i in range(0, rays_o.shape[0], args.chunk_ray):
            # Forward
            with torch.no_grad():
                rays_o_batch = rays_o[i:(i+args.chunk_ray)]
                rays_d_batch = rays_d[i:(i+args.chunk_ray)]
                viewdirs_batch = viewdirs[i:(i+args.chunk_ray)]
                rays = Rays(rays_o_batch, rays_d_batch, viewdirs_batch,
                            args.n_sample_point, args.n_sample_point_fine, near, far, args.perturb)
                ret_dict = nerf_render(rays, point_embedding, view_embedding, model, model_fine,
                                       fine_sampling=fine_sampling,
                                       density_noise_std=0.,
                                       white_bkgd=args.white_bkgd)

                rgb_map.append(ret_dict['rgb_map'])
                if args.n_sample_point_fine > 0:
                    rgb_map_fine.append(ret_dict['rgb_map_fine'])

        rgb_map = torch.cat(rgb_map, 0).reshape(target.shape)
        loss_img = img_loss(rgb_map, target)
        psnr = mse_to_psnr(loss_img.detach().cpu())
        if args.n_sample_point_fine > 0:
            rgb_map_fine = torch.cat(rgb_map_fine, 0).reshape(target.shape)
            loss_img_fine = img_loss(rgb_map_fine, target)
            psnr_fine = mse_to_psnr(loss_img_fine.detach().cpu())
        else:
            loss_img_fine = torch.zeros()
            psnr_fine = torch.zeros()
        loss = loss_img + loss_img_fine

        # Accumulate test results
        eval_meter.append_loss({'mse': loss_img.item(), 'mse_fine': loss_img_fine.item(),
                                'psnr': psnr.item(), 'psnr_fine': psnr_fine.item()})

        # Save rendered images
        rgb_map = rgb_map.cpu().numpy().clip(0., 1.)
        save_path = osp.join(save_render_base, '%06d.png'%(vid))
        imageio.imwrite(save_path, rgb_map)
        rgb_map_fine = rgb_map_fine.cpu().numpy().clip(0., 1.)
        save_path_fine = osp.join(save_render_base, '%06d_fine.png'%(vid))
        imageio.imwrite(save_path_fine, rgb_map_fine)

        tbar.update(1)

    # Log test results
    eval_avg = eval_meter.get_mean_loss_dict()
    print(eval_avg)