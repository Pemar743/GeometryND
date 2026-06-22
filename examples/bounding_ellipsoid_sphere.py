import os
import sys
import time

import numpy as np
import open3d as o3d

from geometryND import SphereND, EllipsoidND
from examples.utils import shape_to_lineset
        

if __name__ == "__main__":

    project_to_xy_plane = True  # Set to True to project the mesh onto the XY plane

    dataset = o3d.data.MonkeyModel()
    # dataset = o3d.data.BunnyMesh()
    # dataset = o3d.data.ArmadilloMesh()
    # dataset = o3d.data.SwordModel()
    # dataset = o3d.data.KnotMesh()
    # dataset = o3d.data.FlightHelmetModel()


    mesh = o3d.io.read_triangle_mesh(dataset.path)

    points = np.asarray(mesh.vertices)
    if project_to_xy_plane:
        points[:, 2] = 0   # set all z-values to zero
        mesh.vertices = o3d.utility.Vector3dVector(points)

    # mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    
    print("Mesh loaded.")

    # -------------------------------------------------------------------------
    # Oriented Bounding Ellipsoid
    # -------------------------------------------------------------------------

    t0 = time.perf_counter()

    pymbe, mbe_boundary = EllipsoidND.minimum_enclosing(points)
    
    t1 = time.perf_counter()
    
    ellipsoid_center = pymbe.center
    ellipsoid_radii = pymbe.radii

    pymbe_time_ms = (t1 - t0) * 1000.0

    print(f"Ellipsoid computed in {pymbe_time_ms:.3f} ms")
    print(f"Ellipsoid center: {ellipsoid_center}")
    print(f"Ellipsoid radii: {ellipsoid_radii}")

    # -------------------------------------------------------------------------
    # Python SphereND
    # -------------------------------------------------------------------------

    t0 = time.perf_counter()

    pymbs, mbs_boundary = SphereND.minimum_enclosing(points)

    t1 = time.perf_counter()

    pybs_time_ms = (t1 - t0) * 1000.0

    print(f"Sphere computed in {pybs_time_ms:.3f} ms")
    print(f"Sphere center: {pymbs.center}")
    print(f"Sphere radius: {pymbs.radius}")

    # -------------------------------------------------------------------------
    # Create Python sphere visualization
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    print("\nExecution Times")
    print("----------------------------")
    print(f"Oriented Bounding Ellipsoid: {pymbe_time_ms:.3f} ms")
    print(f"Exact Sphere:  {pybs_time_ms:.3f} ms")

    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------

    o3d.visualization.draw_geometries([mesh, mbe_ls, mbe_pts, mbs_ls, mbs_pts],
                                      window_name="Bounding Ellipsoid/Sphere", width=800, height=600)