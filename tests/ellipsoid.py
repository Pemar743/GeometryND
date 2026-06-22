from geometryND import EllipsoidND
import numpy as np
import open3d as o3d

if __name__ == "__main__":

    o3d_ellipsoid = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
    o3d_ellipsoid.translate((5.5, 6.5, 7.5))  # Move the ellipsoid to a new center
    
    sx, sy, sz = 1.0, 2.0, 3.0  # Scaling factors for the ellipsoid
    
    S = np.diag([sx, sy, sz, 1.0])  # Scaling matrix for homogeneous coordinates

    center = o3d_ellipsoid.get_center()
    T = np.eye(4)
    T[:3, 3] = -center
    T2 = np.eye(4)
    T2[:3, 3] = center

    o3d_ellipsoid.transform(T2 @ S @ T)  # Apply scaling transformation

    points = np.asarray(o3d_ellipsoid.vertices)

    ####################
    # test for 3D points
    ellipsoid, boundary_points = EllipsoidND.minimum_enclosing(points)

    print("Ellipsoid center:", ellipsoid.center)
    print("Ellipsoid radii:", ellipsoid.radii)

    assert np.allclose(ellipsoid.center, [5.5, 6.5, 7.5]), "Ellipsoid center should be (5.5, 6.5, 7.5)"
    assert np.allclose(ellipsoid.radii, [3.0, 2.0, 1.0]), "Ellipsoid radii should be (3.0, 2.0, 1.0)"
    ####################

    ####################
    # test for degenerate 3D points by projecting onto the XY plane
    points[:, 2] = 0  # Project points onto the XY plane
    ellipsoid, boundary_points = EllipsoidND.minimum_enclosing(points)
    print("Projected Ellipsoid center:", ellipsoid.center)
    print("Projected Ellipsoid radii:", ellipsoid.radii)

    assert np.allclose(ellipsoid.center, [5.5, 6.5, 0.0]), "Projected ellipsoid center should be (5.5, 6.5, 0.0)"
    assert np.allclose(ellipsoid.radii, [0.0, 2.0, 1.0]), "Projected ellipsoid radii should be (0.0, 2.0, 1.0)"
    ####################

    ####################
    # test for 2D points by keeping only the XY coordinates
    points = points[:, :2]  # Keep only the XY coordinates for 2D
    ellipse, boundary_points = EllipsoidND.minimum_enclosing(points)
    print("2D Ellipse center:", ellipse.center)
    print("2D Ellipse radii:", ellipse.radii)

    assert np.allclose(ellipse.center, [5.5, 6.5]), "2D ellipse center should be (5.5, 6.5)"
    assert np.allclose(ellipse.radii, [2.0, 1.0]), "2D ellipse radii should be (2.0, 1.0)"
    ####################