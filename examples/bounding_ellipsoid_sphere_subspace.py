import numpy as np
import open3d as o3d

from geometryND import SphereND, EllipsoidND
from utils import shape_to_lineset

if __name__ == "__main__":

    project_to_xy_plane = True  # Set to True to project the mesh onto the XY plane

    dataset = o3d.data.MonkeyModel()

    mesh = o3d.io.read_triangle_mesh(dataset.path)
    points = np.asarray(mesh.vertices)
    points[:, 2] = 0   # set all z-values to zero
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.compute_vertex_normals()

    pymbe, mbe_boundary = EllipsoidND.minimum_enclosing(points)
    pymbs, mbs_boundary = SphereND.minimum_enclosing(points)

    mbe_ls = shape_to_lineset(pymbe, resolution=20, colour=(0,1,0))
    mbs_ls = shape_to_lineset(pymbs, resolution=20, colour=(0,0,1))

    mbe_boundary = np.asarray(mbe_boundary)
    mbs_boundary = np.asarray(mbs_boundary)

    mbe_pts = o3d.geometry.PointCloud()
    mbe_pts.points = o3d.utility.Vector3dVector(mbe_boundary)
    mbe_pts.paint_uniform_color((0,0.7,0))

    mbs_pts = o3d.geometry.PointCloud()
    mbs_pts.points = o3d.utility.Vector3dVector(mbs_boundary)
    mbs_pts.paint_uniform_color((0,0,0.7))

    o3d.visualization.draw_geometries([mesh, mbe_ls, mbe_pts, mbs_ls, mbs_pts],
                                      window_name="Bounding Ellipsoid/Sphere", width=800, height=600)