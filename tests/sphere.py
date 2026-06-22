from geometryND import SphereND
import numpy as np
import open3d as o3d

if __name__ == "__main__":

    o3d_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
    o3d_sphere.translate((5.5, 6.5, 7.5))  # Move the sphere to a new center
    points = np.asarray(o3d_sphere.vertices)

    ####################
    # test for 3D points
    sphere, boundary_points = SphereND.minimum_enclosing(points)
    
    print("Sphere center:", sphere.center)
    print("Sphere radius:", sphere.radius)

    assert np.allclose(sphere.center, [5.5, 6.5, 7.5]), "Sphere center should be (5.5, 6.5, 7.5)"
    assert np.isclose(sphere.radius, 1.0), "Sphere radius should be 1.0"
    ####################

    ####################
    # test for 2D points by projecting onto the XY plane
    points[:, 2] = 0  # Project points onto the XY plane
    sphere, boundary_points = SphereND.minimum_enclosing(points)
    print("Projected Sphere center:", sphere.center)
    print("Projected Sphere radius:", sphere.radius)

    assert np.allclose(sphere.center, [5.5, 6.5, 0.0]), "Projected sphere center should be (5.5, 6.5, 0.0)"
    assert np.isclose(sphere.radius, 1.0), "Projected sphere radius should be 1.0"
    ####################

    ####################
    # test for 2D points by keeping only the XY coordinates
    points = points[:, :2]  # Keep only the XY coordinates for 2D
    circle, boundary_points = SphereND.minimum_enclosing(points)
    print("2D Circle center:", circle.center)
    print("2D Circle radius:", circle.radius)

    assert np.allclose(circle.center, [5.5, 6.5]), "2D circle center should be (5.5, 6.5)"
    assert np.isclose(circle.radius, 1.0), "2D circle radius should be 1.0"
    ####################