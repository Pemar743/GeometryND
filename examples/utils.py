
import numpy as np
import open3d as o3d
from geometryND import SphereND, EllipsoidND

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