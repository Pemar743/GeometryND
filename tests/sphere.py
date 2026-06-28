from geometryND import SphereND
import numpy as np
import open3d as o3d

if __name__ == "__main__":

    center = np.array([5.5, 6.5, 7.5])
    radius = 1.0
    o3d_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=20)
    o3d_sphere.translate(center)  # Move the sphere to a new center
    points = np.asarray(o3d_sphere.vertices)

    ####################
    # test for 3D points
    min_sphere, boundary_points = SphereND.minimum_enclosing(points)
    
    fit_sphere = SphereND.best_fit(points)

    print(f"{'*'*5} 3D case: {'*'*5}")
    print("Minimum enclosing sphere center:", min_sphere.center)
    print("Minimum enclosing sphere radius:", min_sphere.radius)
    print("Best fit sphere center:", fit_sphere.center)
    print("Best fit sphere radius:", fit_sphere.radius)

    assert np.allclose(min_sphere.center, center), "Minimum enclosing sphere center should be the specified center"
    assert np.isclose(min_sphere.radius, radius), "Minimum enclosing sphere radius should be the specified radius"
    assert np.allclose(fit_sphere.center, center), "Best fit sphere center should be the specified center"
    assert np.isclose(fit_sphere.radius, radius), "Best fit sphere radius should be the specified radius"

    ####################

    ####################
    # test for degenerate 3D points by projecting onto the XY plane
    points_degenerate = points.copy()
    points_degenerate[:, 2] = 0  # Project points onto the XY plane
    center_degenerate = center.copy()
    center_degenerate[2] = 0  # Project center onto the XY plane
    min_sphere_degenerate, boundary_points = SphereND.minimum_enclosing(points_degenerate)

    # Remove interior points that
    # bias the least-squares best-fit for degenerate (flattened) clouds.
    tol_keep = 1e-3
    mask = np.abs(np.linalg.norm(points_degenerate - center_degenerate, axis=1) - radius) <= tol_keep
    fit_sphere_degenerate = SphereND.best_fit(points_degenerate[mask])  # Fit best sphere to the projected points

    print(f"\n{'*'*5} Degenerate case: Projected onto XY plane {'*'*5}")
    print("Projected Minimum Enclosing Sphere center:", min_sphere_degenerate.center)
    print("Projected Minimum Enclosing Sphere radius:", min_sphere_degenerate.radius)
    print("Projected Best Fit Sphere center:", fit_sphere_degenerate.center)
    print("Projected Best Fit Sphere radius:", fit_sphere_degenerate.radius)

    assert np.allclose(min_sphere_degenerate.center, center_degenerate), "Projected minimum enclosing sphere center should be the specified center"
    assert np.isclose(min_sphere_degenerate.radius, radius), "Projected minimum enclosing sphere radius should be the specified radius"
    assert np.allclose(fit_sphere_degenerate.center, center_degenerate), "Projected best fit sphere center should be the specified center"
    assert np.isclose(fit_sphere_degenerate.radius, radius), "Projected best fit sphere radius should be the specified radius"
    ####################

    ####################
    # test for 2D points by keeping only the XY coordinates
    points_2d = points[:, :2]  # Keep only the XY coordinates for 2D
    min_circle, boundary_points = SphereND.minimum_enclosing(points_2d)
    # mask = abs(np.linalg.norm(points_2d - center[:2], axis=1) - radius) <= tol_keep
    fit_circle = SphereND.best_fit(points_2d[mask])  # Fit best circle to the projected points
    
    print(f"\n{'*'*5} 2D case: {'*'*5}")
    print("2D Circle center:", min_circle.center)
    print("2D Circle radius:", min_circle.radius)
    print("2D Best Fit Circle center:", fit_circle.center)
    print("2D Best Fit Circle radius:", fit_circle.radius)

    assert np.allclose(min_circle.center, center[:2]), "2D circle center should be the specified center"
    assert np.isclose(min_circle.radius, radius), "2D circle radius should be the specified radius"
    assert np.allclose(fit_circle.center, center[:2]), "2D best fit circle center should be the specified center"
    assert np.isclose(fit_circle.radius, radius), "2D best fit circle radius should be the specified radius"
    ####################