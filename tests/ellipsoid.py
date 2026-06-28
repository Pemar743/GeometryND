from geometryND import EllipsoidND
import numpy as np
import open3d as o3d

if __name__ == "__main__":

    center = np.array([5.5, 6.5, 7.5])
    radius = 1.0
    radii = np.array([3.0, 2.0, 1.0])  # Radii for the ellipsoid along x, y, z axes
    
    o3d_ellipsoid = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=20)
    o3d_ellipsoid.translate(center)  # Move the ellipsoid to a new center

    points = np.asarray(o3d_ellipsoid.vertices)

    points_degenerate = points.copy()
    points_degenerate[:, 2] = 0  # Project points onto the XY plane
    center_degenerate = center.copy()
    center_degenerate[2] = 0  # Project center onto the XY plane

    tol_keep = 1e-3
    mask = np.abs(np.linalg.norm(points_degenerate - center_degenerate, axis=1) - radius) <= tol_keep
    
    # sx, sy, sz = 1.0, 2.0, 3.0  # Scaling factors for the ellipsoid
    
    S = np.diag([radii[0], radii[1], radii[2], 1.0])  # Scaling matrix for homogeneous coordinates

    T = np.eye(4)
    T[:3, 3] = -center
    T2 = np.eye(4)
    T2[:3, 3] = center

    o3d_ellipsoid.transform(T2 @ S @ T)  # Apply scaling transformation

    points = np.asarray(o3d_ellipsoid.vertices)
    o3d_ellipsoid.compute_vertex_normals()

    points_degenerate = points.copy()
    points_degenerate[:, 2] = 0 #center[2]  # Project points onto the plane z = center[2]
    center_degenerate = center.copy()
    center_degenerate[2] = 0 #center[2]
    radii_degenerate = radii.copy()
    radii_degenerate[2] = 0  # Project radii onto the XY plane

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points_degenerate)
    # pcd.paint_uniform_color([1.0, 0.0, 0.0])  # Color the point cloud red

    # o3d.visualization.draw_geometries([pcd], window_name="Ellipsoid with Point Cloud", width=800, height=600)


    ####################
    # test for 3D points
    min_ellipsoid, boundary_points = EllipsoidND.minimum_enclosing(points)
    fit_ellipsoid = EllipsoidND.best_fit(points)

    print(f"{'*'*5} 3D case: {'*'*5}")
    print("Minimum Ellipsoid center:", min_ellipsoid.center)
    print("Minimum Ellipsoid radii:", min_ellipsoid.radii)
    print("Fitted Ellipsoid center:", fit_ellipsoid.center)
    print("Fitted Ellipsoid radii:", fit_ellipsoid.radii)

    assert np.allclose(min_ellipsoid.center, center), f"Minimum Ellipsoid center should be {center}"
    assert np.allclose(min_ellipsoid.radii, radii), f"Minimum Ellipsoid radii should be {radii}"
    assert np.allclose(fit_ellipsoid.center, center), f"Fitted Ellipsoid center should be {center}"
    assert np.allclose(fit_ellipsoid.radii, radii), f"Fitted Ellipsoid radii should be {radii}"
    ####################

    ####################
    # test for 3D degenerate points
    min_ellipsoid, boundary_points = EllipsoidND.minimum_enclosing(points_degenerate)
    fit_ellipsoid = EllipsoidND.best_fit(points_degenerate[mask])  # Fit best ellipsoid to the projected boundary points

    print(f"\n{'*'*5} Degenerate case: Projected onto XY plane {'*'*5}")
    print("Minimum Degenerate Ellipsoid center:", min_ellipsoid.center)
    print("Minimum Degenerate Ellipsoid radii:", min_ellipsoid.radii)
    print("Fitted Degenerate Ellipsoid center:", fit_ellipsoid.center)
    print("Fitted Degenerate Ellipsoid radii:", fit_ellipsoid.radii)

    assert np.allclose(min_ellipsoid.center, center_degenerate), f"Minimum Degenerate Ellipsoid center should be {center_degenerate}"
    assert np.allclose(min_ellipsoid.radii, radii_degenerate), f"Minimum Degenerate Ellipsoid radii should be {radii_degenerate}"
    assert np.allclose(fit_ellipsoid.center, center_degenerate), f"Fitted Degenerate Ellipsoid center should be {center_degenerate}"
    assert np.allclose(fit_ellipsoid.radii, np.array([2.59807621, 1.73205081, 0.0]), atol=1e-6), "Fitted Degenerate Ellipsoid radii should match the least-squares fit"
    ####################

    ####################
    # test for 2D points by keeping only the XY coordinates
    points_2d = points[:, :2]  # Keep only the XY coordinates for 2D
    min_ellipse_2d, boundary_points = EllipsoidND.minimum_enclosing(points_2d)
    fit_ellipse_2d = EllipsoidND.best_fit(points_2d[mask])  # Fit best ellipse to the projected boundary points
    print(f"\n{'*'*5} 2D case: {'*'*5}")
    print("2D Ellipse center:", min_ellipse_2d.center)
    print("2D Ellipse radii:", min_ellipse_2d.radii)
    print("2D Best Fit Ellipse center:", fit_ellipse_2d.center)
    print("2D Best Fit Ellipse radii:", fit_ellipse_2d.radii)

    assert np.allclose(min_ellipse_2d.center, center[:2]), f"2D ellipse center should be {center[:2]}"
    assert np.allclose(min_ellipse_2d.radii, radii[:2]), f"2D ellipse radii should be {radii[:2]}"
    ####################