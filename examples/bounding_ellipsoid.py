import os
import sys
import time

import numpy as np
import open3d as o3d

import geometryND
from geometryND import SphereND, EllipsoidND


print(geometryND.__version__)

def shape_to_lineset(shape, resolution=20, colour=(0,1,0)):

    # Generate unit sphere points
    n_lat, n_lon = resolution, 2 * resolution
    lat = np.linspace(0, np.pi, n_lat)
    lon = np.linspace(0, 2*np.pi, n_lon)
    unit_sphere_pts = np.array([[np.sin(theta)*np.cos(phi),
                                    np.sin(theta)*np.sin(phi),
                                    np.cos(theta)]
                                for theta in lat for phi in lon])

    C = shape.center

    if isinstance(shape, SphereND): #np.isscalar(RA) or RA.shape == ():  # Sphere
        R = shape.radius
        shape_pts = unit_sphere_pts * R + C
        shape_type = "sphere"
    elif isinstance(shape, EllipsoidND): # and RA.shape == (3,3):  # Ellipsoid
        A = shape.A
        eigvals, eigvecs = np.linalg.eigh(A)
        radii = shape.radii
        shape_pts = (unit_sphere_pts @ np.diag(radii) @ eigvecs.T) + C
        shape_type = "ellipsoid"
    else:
        raise ValueError("RA must be either a float (sphere) or a 3x3 matrix (ellipsoid)")

    # Generate wireframe lines
    lines = []
    for i in range(n_lon):
        for j in range(n_lat-1):
            lines.append([i + j*n_lon, i + (j+1)*n_lon])
    for j in range(n_lat):
        for i in range(n_lon-1):
            lines.append([i + j*n_lon, i+1 + j*n_lon])
        lines.append([n_lon-1 + j*n_lon, j*n_lon])

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(shape_pts)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(np.tile(colour, (len(lines),1)))

    return line_set
        

if __name__ == "__main__":


    dataset = o3d.data.MonkeyModel()
    dataset = o3d.data.BunnyMesh()
    # dataset = o3d.data.ArmadilloMesh()
    # dataset = o3d.data.SwordModel()
    # dataset = o3d.data.KnotMesh()
    # dataset = o3d.data.FlightHelmetModel()


    mesh = o3d.io.read_triangle_mesh(dataset.path)
    # mesh.compute_triangle_normals()
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