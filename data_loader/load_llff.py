import os
import os.path as osp
import numpy as np
import imageio


def _load_data(data_root, factor=None):
    poses_arr = np.load(osp.join(data_root, 'poses_bounds.npy'))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5])
    hwf = poses[:, :, 4]        # (N, 3)
    poses = poses[:, :, :4]     # (N, 3, 4)
    bounds = poses_arr[:, -2:]      # (N, 2)

    sfx = ''
    if factor is not None:
        sfx = '_{}'.format(factor)
    else:
        factor = 1
    imgdir = osp.join(data_root, 'images' + sfx)

    imgfiles = [osp.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if
                f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    assert len(imgfiles) == poses.shape[0], "Number of images and poses must match!"

    img_hw = imageio.imread(imgfiles[0]).shape[:2]
    hwf[:, :2] = img_hw
    hwf[:, 2] = hwf[:, 2] / factor

    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)

    imgs = [imread(f)[..., :3] / 255. for f in imgfiles]
    imgs = np.stack(imgs, 0)
    return imgs.astype(np.float32), poses.astype(np.float32), hwf.astype(np.float32), bounds.astype(np.float32)


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(vec2, up, pos):
    """
    Perform orthogonalization.
    """
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def poses_avg(poses):
    """
    Compute the "average pose" from a set of poses.
    :param poses: (N, 3, 4)
    :return:
        c2w: (3, 4)
    """
    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = viewmatrix(vec2, up, center)
    return c2w


def recenter_poses(poses):
    """
    Re-center a set of poses (place them around the origin and canonical world frame).
    :param poses: (N, 3, 4)
    :return:
        poses: (N, 3, 4)
    """
    # Compute the "average pose"
    bottom = np.reshape([0, 0, 0, 1.], [1, 4]).astype(np.float32)
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], 0)

    # Reduce the "average pose"
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses, bottom], 1)
    poses = np.linalg.inv(c2w) @ poses
    # return poses[:, :3]
    return poses


def load_llff_data(data_root, factor=8, recenter=True, bound_factor=.75):
    images, poses, hwf, bounds = _load_data(data_root, factor=factor)  # factor=8 downsamples original imgs by 8x
    print('Loaded', data_root)
    print('Scene bounds:', bounds.min(), bounds.max())

    # Correct rotation matrix ordering
    poses = np.concatenate([poses[:, :, 1:2], -poses[:, :, 0:1], poses[:, :, 2:]], 2)

    # Rescale if bound_factor is provided
    scale = 1. if bound_factor is None else 1. / (bounds.min() * bound_factor)
    poses[:, :3, 3] *= scale
    bounds *= scale

    # Re-center the poses (place them around the origin and canonical world frame)
    if recenter:
        poses = recenter_poses(poses)

    img_h, img_w, focal = hwf[0]

    return images, poses, bounds, img_h, img_w, focal


if __name__ == '__main__':
    data_root = '/home/ziyang/Desktop/Datasets/nerf_dataset/nerf_llff_data/fern'
    images, poses, bounds, img_h, img_w, focal = load_llff_data(data_root, factor=8, recenter=True, bound_factor=0.75)
    print(images.shape, poses.shape)
    print(images.mean(), images.std())
    print(poses.mean(), poses.std())
    print(img_h, img_w, focal)