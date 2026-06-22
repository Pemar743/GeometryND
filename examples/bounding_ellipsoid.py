import os
import sys
import time

import numpy as np
import open3d as o3d

import geometryND
from geometryND import SphereND, EllipsoidND
from utils import shape_to_lineset

print(geometryND.__version__)

if __name__ == "__main__":


    dataset = o3d.data.MonkeyModel()
    dataset = o3d.data.BunnyMesh()
    # dataset = o3d.data.ArmadilloMesh()
    # dataset = o3d.data.SwordModel()
    # dataset = o3d.data.KnotMesh()
    # dataset = o3d.data.FlightHelmetModel()


    mesh = o3d.io.read_triangle_mesh(dataset.path)
    # mesh.compute_triangle_normals()

    points = np.asarray(mesh.vertices)
    points[:, 2] = 0   # set all z-values to zero
    mesh.vertices = o3d.utility.Vector3dVector(points)

    mesh.compute_vertex_normals()
    
    print("Mesh loaded.")

    points = np.asarray(mesh.vertices)

    # -------------------------------------------------------------------------
    # Oriented Bounding Ellipsoid
    # -------------------------------------------------------------------------

    t0 = time.perf_counter()

    pymbe, mbe_boundary = EllipsoidND.minimum_enclosing(points)
    
    t1 = time.perf_counter()
    
    ellipsoid_center = pymbe.center
    ellipsoid_radii = pymbe.radii

    pymbe_time_ms = (t1 - t0) * 1000.0

    print(f"OBE computed in {pymbe_time_ms:.3f} ms")
    print(f"OBE center: {ellipsoid_center}")
    print(f"OBE radii: {ellipsoid_radii}")

    
    # -------------------------------------------------------------------------
    # Create Python sphere visualization
    # -------------------------------------------------------------------------

    mbe_ls = shape_to_lineset(pymbe, resolution=20, colour=(0,1,0))

    mbe_boundary = np.asarray(mbe_boundary)

    mbe_pts = o3d.geometry.PointCloud()
    mbe_pts.points = o3d.utility.Vector3dVector(mbe_boundary)
    mbe_pts.paint_uniform_color((1,0,0))

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    print("\nExecution Times")
    print("----------------------------")
    print(f"Oriented Bounding Ellipsoid: {pymbe_time_ms:.3f} ms")

    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------

    o3d.visualization.draw_geometries([mesh, mbe_ls, mbe_pts],
                                      window_name="Bounding Ellipsoid", width=800, height=600)