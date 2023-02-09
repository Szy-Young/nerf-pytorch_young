import open3d as o3d


def build_colored_pointcloud(pc, color):
    """
    :param pc: (N, 3).
    :param color: (N, 3).
    """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pc)
    point_cloud.colors = o3d.utility.Vector3dVector(color)
    return point_cloud