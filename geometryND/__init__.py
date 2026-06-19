import sys

from scipy.spatial import ConvexHull
import numpy as np
from numpy.linalg import lstsq, eig, solve
import math
# from functools import cached_property

class EllipsoidND:
    def __init__(self, center: np.ndarray, A: np.ndarray):
        """
        n-dimensional ellipsoid defined by (x - c)^T A (x - c) = 1,
        where A is symmetric positive definite.
        """
        self.center = np.asarray(center, dtype=float)
        self.A = np.asarray(A, dtype=float)

    def __repr__(self):
        shape = "Ellipse" if self.n == 2 else f"Ellipsoid-{self.n}D"
        return f"<{shape}: center={self.center.tolist()}, radii={np.round(self.radii, 3).tolist()}>"
    
    # @cached_property
    def _eigendecomposition(self):
        """Compute and cache eigenvalues/eigenvectors of A."""
        eigvals, eigvecs = np.linalg.eigh(self.A)
        return eigvals, eigvecs

    @property
    def radii(self) -> np.ndarray:
        """Return the semi-axis lengths (radii), setting degenerate axes to 0."""
        eigvals, _ = self._eigendecomposition()
        # Avoid division by zero or negative rounding noise
        with np.errstate(divide='ignore', invalid='ignore'):
            radii = 1.0 / np.sqrt(eigvals)
        # Replace infinities or NaNs (from near-zero or negative eigenvalues) with 0.0
        radii[~np.isfinite(radii)] = 0.0
        return radii

    @property
    def axes(self) -> np.ndarray:
        """Return the principal axes (eigenvectors)."""
        _, eigvecs = self._eigendecomposition()
        return eigvecs

    @property
    def n(self) -> int:
        """Ambient dimension of the ellipsoid."""
        return self.A.shape[0]

    @property
    def volume(self) -> float:
        """Compute the n-dimensional volume of the ellipsoid."""
        n = self.n
        return (math.pi ** (n / 2)) / math.gamma(n / 2 + 1) / np.sqrt(np.linalg.det(self.A))

    def get_residuals(self, X):
        """
        Compute residuals of points relative to the ellipsoid (x-c)^T A (x-c) = 1

        Parameters
        ----------
        X : (N, d) ndarray
            Point coordinates

        Returns
        -------
        r : (N,) ndarray
            Residuals for each point
        """
        c = self.center
        A = self.A
        X_shift = X - c
        r = np.einsum('ij,jk,ik->i', X_shift, A, X_shift) - 1
        return r

    @staticmethod
    def project_to_subspace(points: np.ndarray, tol: float = 1e-9):
        """
        Project a set of points into its intrinsic lower-dimensional affine subspace
        if it is degenerate.

        Parameters
        ----------
        points : (M, D) ndarray
            Input points in N-D space.
        tol : float
            Tolerance for detecting rank deficiency.

        Returns
        -------
        points_proj : (M, r) ndarray
            Points projected into r-dimensional intrinsic subspace.
        basis : (r, n) ndarray
            Orthonormal basis of the intrinsic subspace.
        offset : (D,) ndarray
            Offset used for centering before projection (first point).
        is_in_subspace : bool
            True if the points lie in a lower-dimensional subspace.
        """
        points = np.asarray(points)
        M, D = points.shape

        # Compute rank
        rank = intrinsic_dimension(points, tol=tol, return_centered_pts=False)
        offset = points[0]
        centered = points - offset
        if rank < D:
            # Project into lower-dimensional intrinsic subspace
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            basis = Vt[:rank]  # orthonormal basis of intrinsic subspace
            points_proj = centered @ basis.T
            return points_proj, basis, offset, True

        # Full-rank: return original points with identity basis
        return centered, np.eye(D), offset, False

    # @staticmethod
    # def project_to_subspace(points: np.ndarray, intrinsic_dimension: int, data_centered:bool=True):
    #     """
    #     Project a set of points into its intrinsic lower-dimensional affine subspace
    #     if it is degenerate.
    #
    #     Parameters
    #     ----------
    #     points : (M, D) ndarray
    #         Input points in N-D space.
    #     intrinsic_dimension : float
    #         intrinsic lower-dimensional affine subspace.
    #
    #     Returns
    #     -------
    #     points_proj : (M, r) ndarray
    #         Points projected into r-dimensional intrinsic subspace.
    #     basis : (r, n) ndarray
    #         Orthonormal basis of the intrinsic subspace.
    #     """
    #     points = np.asarray(points)
    #
    #
    #     # Project into lower-dimensional intrinsic subspace
    #     U, S, Vt = np.linalg.svd(points, full_matrices=False)
    #     basis = Vt[:intrinsic_dimension]  # orthonormal basis of intrinsic subspace
    #     points_proj = points @ basis.T
    #     return points_proj, basis


    @staticmethod
    def lift_from_subspace(points_sub, basis, offset):
        """
        Lift points from a lower-dimensional subspace back to the original N-D space.

        Parameters
        ----------
        points_sub : (M, k) or (k,) ndarray
            Points in the k-dimensional subspace (or a single point)
        basis : (k, N) ndarray
            Basis of the subspace (rows are basis vectors)
        offset : (N,) ndarray
            Offset of the subspace origin in N-D space

        Returns
        -------
        points_nd : (M, N) or (N,) ndarray
            Points mapped back to N-D space
        """
        points_sub = np.atleast_2d(points_sub)  # ensure 2D for matrix multiplication
        lifted = offset + points_sub @ basis
        if lifted.shape[0] == 1:
            return lifted[0]  # return 1D array if input was a single point
        return lifted

    # @staticmethod
    # def lift_from_subspace(center_sub, A_sub, basis, offset):
    #     """
    #     Map an ellipsoid fit from a lower-dimensional subspace back to the original N-D space.
    #
    #     Parameters
    #     ----------
    #     center_sub : (k,) ndarray
    #         Center of the ellipsoid in the lower-dimensional subspace.
    #     A_sub : (k, k) ndarray
    #         Shape matrix of the ellipsoid in the subspace.
    #     basis : (k, n) ndarray
    #         Orthonormal basis for the subspace (k < n).
    #     offset : (n,) ndarray
    #         Offset of the subspace in the original space.
    #
    #     Returns
    #     -------
    #     center_nd : (n,) ndarray
    #         Center in the original N-D space.
    #     A_nd : (n, n) ndarray
    #         Shape matrix in the original N-D space.
    #     """
    #     # Map center
    #     center_nd = offset + basis.T @ center_sub
    #
    #     # Map shape matrix
    #     A_nd = basis.T @ A_sub @ basis
    #
    #     return center_nd, A_nd

    @classmethod
    def best_fit(cls, X: np.ndarray, tol: float = 1e-9):
        """
        Fit an n-dimensional ellipsoid to a set of points using least squares,
        automatically handling degenerate (lower-rank) point clouds.
        The constant term c is normalized to 1, reducing the minimum number of
        points required by 1.

        Parameters
        ----------
        X : (N, n) ndarray
            Input point cloud.
        tol : float
            Tolerance for detecting rank deficiency / degeneracy.

        Returns
        -------
        EllipsoidND
            Ellipsoid fitted to the point cloud.
        """
        X = np.asarray(X)
        N, n = X.shape

        # --- Check for points being wholly in lower subspace ---
        X_proj, basis, offset, is_in_subspace = cls.project_to_subspace(X, tol=tol)
        if is_in_subspace:
            # recursively fit in lower-dim space
            ellipsoid_sub = cls.best_fit(X_proj)
            # Map center back to original N-D
            center_nd = cls.lift_from_subspace(ellipsoid_sub.center, basis, offset)
            # Map shape matrix back
            A_nd = basis.T @ ellipsoid_sub.A @ basis
            return cls(center_nd, A_nd)

        # --- Full-rank case ---
        # Minimum points required to fit with c normalized to 1
        num_coeffs = n * (n + 1) // 2 + n
        if N < num_coeffs:
            raise ValueError(
                f"Not enough points to fit {n}-D ellipsoid; "
                f"need at least {num_coeffs}, got {N}"
            )

        # --- Build quadratic term matrix ---
        i_idx, j_idx = np.triu_indices(n)
        Q = X[:, i_idx] * X[:, j_idx]  # Quadratic terms
        L = X  # Linear terms
        D = np.hstack([Q, L])  # c is fixed to 1
        rhs = np.ones(N)  # normalized

        # Solve least squares
        coeffs, residuals, rank_ls, s = np.linalg.lstsq(D, rhs, rcond=None)

        # Extract A
        A = np.zeros((n, n))
        A[i_idx, j_idx] = coeffs[:len(i_idx)]
        A[j_idx, i_idx] = coeffs[:len(i_idx)]  # symmetric
        b = coeffs[len(i_idx):]

        # --- Center of ellipsoid ---
        center = -0.5 * np.linalg.solve(A, b)

        # --- Shape matrix normalized so that (x - center)^T A (x - center) = 1 ---
        A_centered = A / (1 - center.T @ A @ center - b.T @ center - 1)  # c=1

        return cls(center, A_centered)

    @classmethod
    def from_points_ransac(cls, X, max_iter=100, tol=1e-3):
        """
        Fit an nD ellipsoid using RANSAC to handle outliers.
        Normalizes the constant term to 1.

        Parameters
        ----------
        X : (N, n) ndarray
            Input points
        max_iter : int
            Maximum RANSAC iterations
        tol : float
            Distance tolerance to count inliers

        Returns
        -------
        EllipsoidND
            Fitted ellipsoid from inliers
        """
        X = np.asarray(X)
        N, n = X.shape
        best_inliers = []
        best_model = None

        # Minimum points required to fit (normalized constant)
        min_pts = n * (n + 1) // 2 + n

        if N < min_pts:
            raise ValueError(f"Need at least {min_pts} points to fit {n}-D ellipsoid.")

        for _ in range(max_iter):
            # Random subset
            idx = np.random.choice(N, min_pts, replace=False)
            subset = X[idx]

            try:
                model = cls.best_fit(subset)
            except np.linalg.LinAlgError:
                continue

            # Compute residuals
            diffs = X - model.center
            res = np.einsum('ij,jk,ik->i', diffs, model.A, diffs) - 1
            inliers = np.where(np.abs(res) <= tol)[0]

            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_model = model

        if best_model is None:
            raise RuntimeError("RANSAC failed to find a valid ellipsoid.")

        # Refit using all inliers for best estimate
        return cls.best_fit(X[best_inliers])

    @classmethod
    def minimum_enclosing(cls,
            X: np.ndarray,
            tol: float = 1e-9,
            max_iter: int = 1000,
            weight_thresh: float = 0.95,
            return_hull_pts: bool = False,
            return_weights: bool = False
    ):
        """
        Compute the minimum-volume enclosing ellipsoid of a point cloud in N dimensions
        using Khachiyan's algorithm, handling degenerate (lower-rank) point clouds.

        Parameters
        ----------
        X : (N_points, N_dims) ndarray
            Input point cloud in N-dimensional space.
        tol : float
            Convergence tolerance for Khachiyan's algorithm.
        max_iter : int
            Maximum number of iterations.
        weight_thresh : float
            Fraction of cumulative sorted weights used to identify controlling points.
            Set to 0 to skip boundary extraction.
        return_hull_pts : bool
            If True, also return the convex hull points used internally.
        return_weights : bool
            If True, also return the weight vector for all hull points.

        Returns
        -------
        ellipsoid : EllipsoidND
            Minimal enclosing ellipsoid (center and shape matrix A).
        Xb : (M, N) ndarray, optional
            Boundary points contributing to weight_thresh fraction.
        X_hull : (K, N) ndarray, optional
            Convex hull points used internally.
        u : (K,) ndarray, optional
            Final weights assigned to each convex hull point.
        """
        X = np.asarray(X)
        N_points, D = X.shape
        if N_points < 1:
            raise ValueError("Input point cloud is empty.")

        # --- Check for degenerate subspace ---
        # Project points to intrinsic lower-dimensional subspace if degenerate
        # Compute rank
        # rank, centered, offset = intrinsic_dimension(X, tol=tol, return_centered_pts=True)

        # if rank < D:
            X_sub, basis = cls.project_to_subspace(centered, intrinsic_dimension=rank)

        X_sub, basis, offset, is_in_subspace = cls.project_to_subspace(X, tol=tol)
        if is_in_subspace:
            # Recursively compute ellipsoid in lower-dimensional space
            ellipsoid_sub, *extra = cls.minimum_enclosing(
                X_sub, tol=tol, max_iter=max_iter,
                weight_thresh=weight_thresh,
                return_hull_pts=return_hull_pts,
                return_weights=return_weights
            )

            # Map center back to original N-D
            center_nd = cls.lift_from_subspace(ellipsoid_sub.center, basis, offset)
            # Map shape matrix back
            A_nd = basis.T @ ellipsoid_sub.A @ basis
            ellipsoid = cls(center_nd, A_nd)

            # Map extra outputs (boundary/hull points) back to N-D
            mapped_extra = []
            for arr in extra:
                if arr is None:
                    mapped_extra.append(None)
                elif return_weights and arr.ndim == 1:
                    # Weight vector, leave as-is
                    mapped_extra.append(arr)
                else:
                    # Array of points, map back
                    mapped_extra.append(offset + (basis.T @ arr.T).T)

            return ellipsoid, *mapped_extra

        # --- Full-rank case ---
        # Reduce points via convex hull for numerical stability if more than D+1 points
        if N_points > D + 1:
            hull = ConvexHull(X)
            X = X[hull.vertices]
            N_points = len(X)

        # Khachiyan initialization
        Q = np.column_stack((X, np.ones(N_points)))  # augment for affine coordinates
        u = np.ones(N_points) / N_points

        for _ in range(max_iter):
            V = Q.T @ np.diag(u) @ Q
            M = np.einsum('ij,jk,ki->i', Q, np.linalg.inv(V), Q.T)
            j = np.argmax(M)
            max_M = M[j]
            step_size = (max_M - D - 1) / ((D + 1) * (max_M - 1))
            new_u = (1 - step_size) * u
            new_u[j] += step_size
            if np.linalg.norm(new_u - u) < tol:
                u = new_u
                break
            u = new_u

        # Compute ellipsoid center and shape matrix
        c = X.T @ u
        X_shift = X - c
        A = np.linalg.inv(X_shift.T @ np.diag(u) @ X_shift) / D

        return_object = [cls(c, A)]

        # Extract boundary points if requested
        if weight_thresh > 0:
            sorted_indices = np.argsort(u)[::-1]
            sorted_weights = u[sorted_indices]
            cumulative_sum_weights = np.cumsum(sorted_weights)
            cutoff_idx = np.searchsorted(cumulative_sum_weights, weight_thresh)
            controlling_point_indices = sorted_indices[:cutoff_idx + 1]
            # fallback if not enough points
            if controlling_point_indices.shape[0] < D + 1:
                controlling_point_indices = sorted_indices[:D + 1]
            Xb = X[controlling_point_indices]
            weights_b = u[controlling_point_indices]
            # sort Xb by their weights descending
            sort_order = np.argsort(weights_b)[::-1]
            Xb = Xb[sort_order]

            return_object.append(Xb)

        if return_hull_pts:
            return_object.append(X)
        if return_weights:
            return_object.append(u)

        return tuple(return_object)
    
# ------------------------------------------------------------
# SphereND as a special case of EllipsoidND
# ------------------------------------------------------------
class SphereND(EllipsoidND):
    def __init__(self, center: np.ndarray, radius: float):
        """
        n-dimensional sphere (special case of EllipsoidND) with all axes equal.
        """
        center = np.asarray(center, dtype=float)
        n = center.size
        # Shape matrix for a sphere: A = I / r^2
        A = np.eye(n) / (radius ** 2)
        super().__init__(center, A)
        self.radius = float(radius)  # keep radius for convenience

    def __repr__(self):
        shape = "Circle" if self.n == 2 else f"Sphere-{self.n}D"
        return f"<{shape}: center={self.center.tolist()}, radius={self.radius}>"

    @property
    def radii(self) -> np.ndarray:
        """Return all semi-axes as equal to the radius (fast, no eigen-decomposition)."""
        return np.full(self.n ,self.radius)

    @property
    def surface_area(self) -> float:
        """Compute the (n-1)-dimensional surface area of the sphere."""
        n = self.n
        r = self.radius
        return (2 * math.pi ** (n / 2)) / math.gamma(n / 2) * r ** (n - 1)

    def get_residuals(self, X):
        """
        Compute residuals of points relative to an n-dimensional sphere ||x - c||^2 = r^2

        Parameters
        ----------
        X : (N, d) ndarray
            Point coordinates

        Returns
        -------
        r : (N,) ndarray
            Residuals for each point
        """
        c = self.center
        r = self.radius
        X_shift = X - c
        dist_sq = np.sum(X_shift ** 2, axis=1)
        return dist_sq - r ** 2

    # ------------------------------------------------------------------------
    # Base case: compute minimal circumsphere for ≤ D+1 points
    # ------------------------------------------------------------------------
    @staticmethod
    def fit_circumsphere_nd(points, tol=1e-9):
        """
        Fit the exact minimal circumsphere to ≤ D+1 points in N-D.

        Parameters
        ----------
        points : (M, D) ndarray
            Array of points, M <= D+1

        Returns
        -------
        radius : float
        center : (D,) ndarray
        """
        points = np.asarray(points)
        M, D = points.shape

        if M == 0:
            return np.nan, np.full(D, np.nan)
        if M == 1:
            return 0.0, points[0]
        if M == 2:
            c = points.mean(axis=0)
            r = np.linalg.norm(points[0] - c)
            return r, c

        # ------------------------------------------------------------------------
        # Check for degeneracy: not enough independent points to define a unique sphere
        # ------------------------------------------------------------------------
        rank = intrinsic_dimension(points, tol=tol)

        if rank < M - 1:
            # Degenerate: no unique sphere possible
            return np.inf, np.full(D, np.nan)

        # ------------------------------------------------------------------------
        # Project points into intrinsic lower-dimensional subspace if rank < D
        # ------------------------------------------------------------------------
        if rank < D:
            X_sub, basis, offset, _ = SphereND.project_to_subspace(points, tol=tol)
        # if is_in_subspace:
            # Recursively fit in lower-dimensional space
            r_sub, c_sub = SphereND.fit_circumsphere_nd(X_sub, tol=tol)
            # Lift center back to original space
            center_nd = SphereND.lift_from_subspace(c_sub, basis, offset)
            return r_sub, center_nd

        # ------------------------------------------------------------------------
        # Full-dimensional solve (standard linear system)
        # Solve: 2*(p_i - p_0) · c = |p_i|^2 - |p_0|^2 for i = 1..M-1
        # ------------------------------------------------------------------------
        V = points[1:] - points[0]
        A = 2 * V
        b = np.sum(points[1:] ** 2 - points[0] ** 2, axis=1)
        try:
            center = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            center = np.linalg.lstsq(A, b, rcond=None)[0]

        radius = np.linalg.norm(points[0] - center)
        return radius, center

    @classmethod
    def minimum_enclosing(cls, X, eps=1e-12):
        """
        Compute the exact minimum bounding sphere in N-dimensional space.
    
        Handles degenerate cases automatically:
        - Collinear points
        - Coplanar points
        - Any lower-dimensional subspace
    
        Returns a valid sphere in the intrinsic dimension of the points.
    
        Parameters
        ----------
        X : (M, N) ndarray
            Input points (M points in N dimensions)
        eps : float
            Numerical tolerance for inclusion tests
    
        Returns
        -------
        sphere : SphereND
            Minimal bounding sphere (center and radius)
        boundary_pts : (k, N) ndarray
            Points on the boundary of the sphere
        """

        # ------------------------------------------------------------------------
        # Welzl's algorithm for N-D minimal enclosing sphere
        # ------------------------------------------------------------------------
        def _welzl_exact_minimum_enclosing_nd_sphere(P, B=None, n=None, eps=1e-12):
            """
            Compute the minimal enclosing sphere in N-D using Welzl's algorithm.

            Parameters
            ----------
            P : (M, D) ndarray
                Input points
            B : (<= D+1, D) ndarray
                Boundary points
            n : int
                Number of points from P to consider (used internally)
            eps : float
                Tolerance for inclusion

            Returns
            -------
            radius : float
            center : (D,) ndarray
            boundary_points : ndarray
            """
            P = np.asarray(P)
            M, D = P.shape

            if B is None:
                B = np.empty((0, D))
            if n is None:
                n = M

            max_boundary = D + 1
            if len(B) == max_boundary or n == 0:
                r, c = cls.fit_circumsphere_nd(B)
                return r, c, np.array(B)

            # Recursively exclude last point
            p = P[n - 1]
            r, c, B_current = _welzl_exact_minimum_enclosing_nd_sphere(P, B=B, n=n - 1, eps=eps)

            # If p inside sphere, no change
            if np.linalg.norm(p - c) <= r + eps:
                return r, c, B_current
            else:
                # p must be on boundary
                B_new = np.vstack([B, p])
                return _welzl_exact_minimum_enclosing_nd_sphere(P, B=B_new, n=n - 1, eps=eps)

        def _return_(center, radius, Xb):
            Xb = Xb[np.lexsort(Xb.T[::-1])]  # sort lexicographically
            return cls(center=center, radius=radius), Xb

        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("Input X must be a 2D array of shape (M, N).")
    
        M, D = X.shape
        if M == 0:
            raise ValueError("No points provided.")
    
        # --- Check if points lie in a lower-dimensional subspace ---
        X_sub, basis, offset, is_in_subspace = cls.project_to_subspace(X)
        if is_in_subspace:
            r_sub, c_sub, boundary_sub = _welzl_exact_minimum_enclosing_nd_sphere(X_sub, eps=eps)
            # center_nd, A_nd = cls.lift_from_subspace(c_sub, r_sub, basis, offset)
            # Map center and boundary points back to original space
            center_nd = cls.lift_from_subspace(c_sub, basis, offset)
            Xb_nd = cls.lift_from_subspace(boundary_sub, basis, offset)

            return _return_(center=center_nd, radius=r_sub, Xb=Xb_nd)

        # X_centered = X - X[0]
        # rank = np.linalg.matrix_rank(X_centered, tol=eps)
        #
        # if rank < D:
        #     # Points are degenerate → project to intrinsic subspace
        #     U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        #     V_sub = Vt[:rank]  # basis of intrinsic subspace
        #     X_sub = X_centered @ V_sub.T  # project onto subspace
        #
        #     # Minimal enclosing sphere in lower-dimensional space
        #     r_sub, c_sub, boundary_sub = _welzl_exact_minimum_enclosing_nd_sphere(X_sub, eps=eps)
        #
        #     # Map center and boundary points back to original space
        #     C_nd = X[0] + V_sub.T @ c_sub
        #     B_nd = X[0] + (V_sub.T @ boundary_sub.T).T
        #
        #     # sphere = cls(center=C_nd, radius=r_sub)
        #     return _return_(center=C_nd, radius=r_sub, Xb=B_nd)
    
        # --- Full-dimensional case ---
        # Reduce points via convex hull to speed up
        if M > D + 1:
            hull = ConvexHull(X)
            X = X[hull.vertices]
            X = np.unique(X, axis=0)
    
        # --- Randomly permute points for Welzl recursion ----------------------
        X = np.random.permutation(X)

        # Safe recursion
        if len(X) < int(sys.getrecursionlimit() * 0.95):
            r, c, Xb = _welzl_exact_minimum_enclosing_nd_sphere(X)
            return _return_(center=c, radius=r, Xb=Xb)

        # --- Chunked fallback for large point clouds --------------------------
        M = len(X)
        dM = min(M // 4, 300)
        X_splits = np.array_split(X, dM)
        Xb = np.empty((0, D))
        r, c = np.nan, np.full(D, np.nan)
        for Xi in X_splits:
            Xi = np.concatenate([Xi, Xb], axis=0) if len(Xb) > 0 else Xi
            r, c, Xb = _welzl_exact_minimum_enclosing_nd_sphere(Xi)

        return _return_(center=c, radius=r, Xb=Xb)

# -------------------------------------------------------------------------
# Helper functions for geometric checks
# -------------------------------------------------------------------------
def is_in_k_subspace(points, k, tol=1e-6):
    """
    Check if a set of points lies in a k-dimensional affine subspace in N-D space.

    Parameters
    ----------
    points : array-like, shape (M, N)
        List of M points in N-dimensional space.
    k : int
        Dimension of the subspace (1=line, 2=plane, etc.).
    tol : float
        Tolerance for linear dependence (smaller = stricter).

    Returns
    -------
    bool
        True if points lie in a k-dimensional subspace, False otherwise.
    """
    points = np.asarray(points)
    return intrinsic_dimension(points, tol=tol) <= k

def intrinsic_dimension(points, tol=1e-9, return_centered_pts=False):
    """
    Return the intrinsic affine dimension spanned by a set of points.
    For example:
      - 0: single point
      - 1: colinear (line)
      - 2: coplanar (plane)
      - 3: volume, etc.
    """
    points = np.asarray(points)
    M, D = points.shape

    if M <= 1:
        return 0
    if M == 2:
        return 1

    # translate to origin of affine subspace
    origin = points[0]
    centered = points - origin

    rank = np.linalg.matrix_rank(centered, tol=tol)
    # if return_centered_pts:
    #     return rank, centered, origin
    return rank


# ----------------------------
# Convenience wrappers
# ----------------------------

def is_collinear(points, tol=1e-6):
    """Check if the first three points are nearly collinear in N-D."""
    return is_in_k_subspace(points, k=1, tol=tol)

def is_coplanar(points, tol=1e-6):
    """Check if the first four points are nearly coplanar in N-D."""
    return is_in_k_subspace(points, k=2, tol=tol)



# def is_collinear(points, tol_deg=0.01):
#     """Check if the first three points are nearly collinear."""
#     if len(points) < 3:
#         return False
#     D12 = points[1] - points[0]
#     D13 = points[2] - points[0]
#     D12 /= np.linalg.norm(D12)
#     D13 /= np.linalg.norm(D13)
#     angle = np.degrees(np.arccos(np.clip(np.abs(np.dot(D12, D13)), 0., 1.)))
#     return angle < tol_deg
#
#
# def is_coplanar(points, tol_deg=0.01):
#     """Check if the first four points are nearly coplanar."""
#     if len(points) < 4:
#         return False
#     D12 = points[1] - points[0]
#     D13 = points[2] - points[0]
#     D14 = points[3] - points[0]
#     n1 = np.cross(D12, D13)
#     n2 = np.cross(D12, D14)
#     n1 /= np.linalg.norm(n1)
#     n2 /= np.linalg.norm(n2)
#     angle = np.degrees(np.arccos(np.clip(np.abs(np.dot(n1, n2)), 0., 1.)))
#     return angle < tol_deg


# -------------------------------------------------------------------------
# Fit a circle to ≤3 2D points or a sphere to ≤4 3D points
# -------------------------------------------------------------------------
# def fit_circle_or_sphere_to_points(array, eps_angle=0.01):
#     """
#     Fit a minimal circle (2D) or sphere (3D) to ≤ D+1 points.
#
#     Handles:
#         - 2D: N ≤ 3 points
#         - 3D: N ≤ 4 points
#
#     For 3D with 3 points, returns a circle in the plane defined by the points.
#     For 3D with 4 points, returns a full sphere if not coplanar.
#
#     Parameters
#     ----------
#     array : (N, D) ndarray
#         Point coordinates, where D ∈ {2, 3}, N ≤ D+1.
#     eps_angle : float
#         Angular threshold (degrees) for collinearity/coplanarity detection.
#
#     Returns
#     -------
#     R : float
#         Radius of the enclosing shape (circle or sphere).
#     C : (D,) ndarray
#         Center of the enclosing shape.
#     """
#
#     array = np.asarray(array)
#     N = len(array)
#     D = array.shape[1] if array.ndim > 1 else 2  # default 2 if not known
#
#     if N == 0:
#         return np.nan, np.full(D, np.nan)
#     if N == 1:
#         return 0.0, array[0]
#
#     # D = array.shape[1]
#
#     if D not in (2, 3):
#         raise ValueError("Only 2D and 3D points are supported.")
#     if N > D + 1:
#         raise ValueError(f"{D}D: must have ≤ {D + 1} points.")
#     # Remove duplicates
#     uniq, index = np.unique(array, axis=0, return_index=True)
#     array = uniq[index.argsort()]
#     N = len(array)
#
#     # --- Simple line case (shared by 2D and 3D) ---------------------------
#     if N == 2:
#         R = np.linalg.norm(array[1] - array[0]) / 2
#         C = np.mean(array, axis=0)
#         return R, C
#
#     # --- Global collinearity check for N >= 3 -----------------------------
#     if is_collinear(array, eps_angle):
#         return np.inf, np.full(D, np.nan)
#
#     # --- 3D special case --------------------------------------------------
#     if D == 3:
#
#         # 3 points in 3D → fit circle in plane
#         if N == 3:
#             # Plane normal
#             n = np.cross(array[1] - array[0], array[2] - array[0])
#             n /= np.linalg.norm(n)
#
#             # Align plane to XY using Rodrigues’ rotation formula
#             z = np.array([0, 0, 1])
#             r = np.cross(n, z)
#             if np.linalg.norm(r) < 1e-12:
#                 Rmat = np.eye(3)
#             else:
#                 theta = np.arccos(np.clip(np.dot(n, z), -1, 1))
#                 K = np.array([
#                     [0, -r[2], r[1]],
#                     [r[2], 0, -r[0]],
#                     [-r[1], r[0], 0]
#                 ]) / np.linalg.norm(r)
#                 Rmat = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
#
#             Xr = (Rmat @ array.T).T
#             x2d = Xr[:, :2]
#             A = 2 * (x2d[1:] - x2d[0])
#             b = np.sum(x2d[1:] ** 2 - x2d[0] ** 2, axis=1)
#             try:
#                 C2 = np.linalg.solve(A, b)
#             except np.linalg.LinAlgError:
#                 return np.inf, np.full(3, np.nan)
#             R = np.linalg.norm(x2d[0] - C2)
#
#             # Lift back to 3D
#             C3 = np.append(C2, np.mean(Xr[:, 2]))
#             C3 = Rmat.T @ C3
#             return R, C3
#
#         # 4 points in 3D → check coplanarity
#         if N == 4 and is_coplanar(array, eps_angle):
#             return np.inf, np.full(3, np.nan)
#
#     # --- General linear solve for circle (2D) or sphere (3D) --------------
#     # R^2 = ||pi - C||^2 (where pi is the ith point and ||X|| is the L2 norm of X)
#     # expanding for first point
#     # ||pi - C||^2 - ||p0 - C||^2 = 0
#     # (pi - p0)C = 1/2(||pi||^2 - ||p0||^2)
#
#     # coefficient matrix where AC=b
#     A = 2 * (array[1:] - array[0])
#     b = np.sum(array[1:] ** 2 - array[0] ** 2, axis=1)
#     try:
#         C = np.linalg.solve(A, b)
#     except np.linalg.LinAlgError:
#         return np.inf, np.full(D, np.nan)
#     R = np.linalg.norm(array[0] - C)
#     return R, C
#
# # -------------------------------------------------------------------------
# # Exact minimum bounding circle or sphere (Welzl algorithm)
# # -------------------------------------------------------------------------
# def B_min_enclosing_circle_or_sphere(P, B=None, eps=1e-6, n=None):
#     """
#     Generalized Welzl algorithm for computing the minimum enclosing
#     circle (2D) or sphere (3D), depending on point dimensionality.
#
#     Parameters
#     ----------
#     P : (N, D) array-like
#         Input points, where D is dimensionality (2 or 3).
#     B : list of boundary points (optional: default = None)
#     n : int, number of points from P to consider (used internally)
#     eps : float (optional: default = 1e-6)
#         Small tolerance for inclusion test.
#
#     Returns
#     -------
#     radius : float
#     centre : ndarray, shape (D,)
#     boundary_pts : ndarray, shape (k, D)
#     """
#     P = np.asarray(P)
#     N, D = P.shape
#
#     if B is None:
#         # Create empty 2D array with correct number of columns
#         B = np.empty((0, D))
#     if n is None:
#         n = N
#
#     if len(P) == 0 and len(B) == 0:
#         raise ValueError("Cannot compute bounding shape for empty input.")
#
#     max_boundary = D + 1  # D+1 points define circle/sphere in D dimensions
#
#     # Base case: boundary is full or no points left
#     if len(B) == max_boundary or n == 0:
#         radius, centre = fit_circle_or_sphere_to_points(B)
#         return radius, centre, np.array(B)
#
#     # Recursive case: remove the last point
#     p = P[n-1]
#     radius, centre, boundary_pts = B_min_enclosing_circle_or_sphere(P, B=B, n=n - 1, eps=eps)
#
#     # Check if p is inside the current shape
#     if np.linalg.norm(p - centre) <= radius + eps:
#         return radius, centre, boundary_pts
#     else:
#         B_new = np.vstack([B, p])
#         return B_min_enclosing_circle_or_sphere(P, B=B_new, n=n - 1, eps=eps)
#
#
# # ------------------------------------------------------------------------
# # Base case: compute minimal circumsphere for ≤ D+1 points
# # ------------------------------------------------------------------------
# def fit_circumsphere_nd(points):
#     """
#     Fit the exact minimal circumsphere to ≤ D+1 points in N-D.
#
#     Parameters
#     ----------
#     points : (M, D) ndarray
#         Array of points, M <= D+1
#
#     Returns
#     -------
#     radius : float
#     center : (D,) ndarray
#     """
#     points = np.asarray(points)
#     M, D = points.shape
#
#     if M == 0:
#         return np.nan, np.full(D, np.nan)
#     if M == 1:
#         return 0.0, points[0]
#     if M == 2:
#         c = points.mean(axis=0)
#         r = np.linalg.norm(points[0] - c)
#         return r, c
#
#     V = points[1:] - points[0]
#     rank = np.linalg.matrix_rank(V)
#
#     if rank < M - 1:
#         # Degenerate: no unique sphere possible
#         return np.inf, np.full(D, np.nan)
#
#     if rank < D:
#         # Project to intrinsic subspace
#         U, S, Vt = np.linalg.svd(V, full_matrices=False)
#         basis = Vt[:rank]  # shape (rank, D)
#
#         # Project points into subspace coordinates
#         proj_points = (points - points[0]) @ basis.T  # shape (M, rank)
#
#         # Recursively fit sphere in lower dimension
#         r_sub, c_sub = fit_circumsphere_nd(proj_points)
#
#         # Map center back to original space
#         center = points[0] + basis.T @ c_sub
#         return r_sub, center
#
#     # Full-dimensional solve as before
#     A = 2 * V
#     b = np.sum(points[1:]**2 - points[0]**2, axis=1)
#     try:
#         c = np.linalg.solve(A, b)
#     except np.linalg.LinAlgError:
#         c = np.linalg.lstsq(A, b, rcond=None)[0]
#     r = np.linalg.norm(points[0] - c)
#     return r, c
#
#
#
# # ------------------------------------------------------------------------
# # Welzl's algorithm for N-D minimal enclosing sphere
# # ------------------------------------------------------------------------
# def min_enclosing_sphere_nd(P, B=None, n=None, eps=1e-12):
#     """
#     Compute the minimal enclosing sphere in N-D using Welzl's algorithm.
#
#     Parameters
#     ----------
#     P : (M, D) ndarray
#         Input points
#     B : (<= D+1, D) ndarray
#         Boundary points
#     n : int
#         Number of points from P to consider (used internally)
#     eps : float
#         Tolerance for inclusion
#
#     Returns
#     -------
#     radius : float
#     center : (D,) ndarray
#     boundary_points : ndarray
#     """
#     P = np.asarray(P)
#     M, D = P.shape
#
#     if B is None:
#         B = np.empty((0, D))
#     if n is None:
#         n = M
#
#     max_boundary = D + 1
#     if len(B) == max_boundary or n == 0:
#         r, c = fit_circumsphere_nd(B)
#         return r, c, np.array(B)
#
#     # Recursively exclude last point
#     p = P[n-1]
#     r, c, B_current = min_enclosing_sphere_nd(P, B=B, n=n-1, eps=eps)
#
#     # If p inside sphere, no change
#     if np.linalg.norm(p - c) <= r + eps:
#         return r, c, B_current
#     else:
#         # p must be on boundary
#         B_new = np.vstack([B, p])
#         return min_enclosing_sphere_nd(P, B=B_new, n=n-1, eps=eps)
#
#
# # -------------------------------------------------------------------------
# # exact minimum bounding circle or sphere
# # -------------------------------------------------------------------------
# def exact_min_bounding_sphere_nd(X, eps=1e-12):
#     """
#     Compute the exact minimum bounding sphere in N-dimensional space.
#
#     Handles degenerate cases automatically:
#     - Collinear points
#     - Coplanar points
#     - Any lower-dimensional subspace
#
#     Returns a valid sphere in the intrinsic dimension of the points.
#
#     Parameters
#     ----------
#     X : (M, N) ndarray
#         Input points (M points in N dimensions)
#     eps : float
#         Numerical tolerance for inclusion tests
#
#     Returns
#     -------
#     sphere : SphereND
#         Minimal bounding sphere (center and radius)
#     boundary_pts : (k, N) ndarray
#         Points on the boundary of the sphere
#     """
#     X = np.asarray(X)
#     if X.ndim != 2:
#         raise ValueError("Input X must be a 2D array of shape (M, N).")
#
#     M, D = X.shape
#     if M == 0:
#         raise ValueError("No points provided.")
#
#     # --- Check if points lie in a lower-dimensional subspace ---
#     X_centered = X - X[0]
#     rank = np.linalg.matrix_rank(X_centered, tol=eps)
#
#     if rank < D:
#         # Points are degenerate → project to intrinsic subspace
#         U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
#         V_sub = Vt[:rank]  # basis of intrinsic subspace
#         X_sub = X_centered @ V_sub.T  # project onto subspace
#
#         # Minimal enclosing sphere in lower-dimensional space
#         r_sub, c_sub, boundary_sub = min_enclosing_sphere_nd(X_sub, eps=eps)
#
#         # Map center and boundary points back to original space
#         C_nd = X[0] + V_sub.T @ c_sub
#         B_nd = X[0] + (V_sub.T @ boundary_sub.T).T
#
#         sphere = SphereND(center=C_nd, radius=r_sub)
#         return sphere, B_nd
#
#     # --- Full-dimensional case ---
#     # Reduce points via convex hull to speed up
#     if M > D + 1:
#         hull = ConvexHull(X)
#         X = X[hull.vertices]
#         X = np.unique(X, axis=0)
#
#     # --- Randomly permute points for Welzl recursion ----------------------
#     X = np.random.permutation(X)
#
#     # --- Compute minimal enclosing sphere in full space ---
#     r, c, boundary = min_enclosing_sphere_nd(X, eps=eps)
#     sphere = SphereND(center=c, radius=r)
#     Xb = boundary[np.lexsort(boundary.T[::-1])]
#
#     return sphere, Xb
#
#
# def exact_min_bounding_n_sphere(X, eps_angle=0.01):
#     """
#     Compute the exact minimum bounding circle (2D) or sphere (3D) of a point cloud
#     using Welzl's algorithm with analytic solutions for small point sets.
#
#     The function automatically determines whether to compute a circle or sphere
#     based on the dimensionality of the input data (D = 2 or 3). Degenerate cases
#     such as collinear (2D) or coplanar (3D) points are handled analytically.
#
#     Parameters
#     ----------
#     X : (M, D) ndarray
#         Input point coordinates, where D must be 2 or 3.
#     eps_angle : float, optional
#         Small angular threshold (in degrees) used to detect collinear or coplanar
#         point sets. Default is 0.01.
#
#     Returns
#     -------
#     SphereND
#         A `SphereND` instance representing the minimum bounding circle or sphere,
#         with attributes:
#             - `center` : (D,) ndarray — the center coordinates.
#             - `radius` : float — the radius of the bounding shape.
#     Xb : (k, D) ndarray
#         Boundary points that define the minimal enclosing circle or sphere.
#
#     Notes
#     -----
#     - For small point sets (≤ D+1), an analytic fit is used.
#     - For larger point clouds, the algorithm employs a randomized Welzl recursion
#       with convex hull pre-reduction.
#     - Degenerate cases (e.g., all points on a line or plane) are projected to a
#       lower-dimensional subspace before computing the bounding primitive.
#
#     """
#
#     X = np.asarray(X)
#     if X.ndim != 2 or X.shape[1] not in (2, 3):
#         raise ValueError("Input must be (M, 2) or (M, 3) array.")
#
#     D = X.shape[1]
#
#     # --- Handle small number of points analytically ------------------------
#     if len(X) <= D + 1:
#         R, C = fit_circle_or_sphere_to_points(X, eps_angle)
#         return SphereND(C, R), X
#
#     # --- Check for collinear / coplanar sets using helper functions --------
#     if D == 2 and is_collinear(X, eps_angle):
#         # Project onto line between extreme points
#         line_vec = X[1] - X[0]
#         line_vec /= np.linalg.norm(line_vec)
#         dx = (X - X.mean(axis=0)) @ line_vec
#         id_min, id_max = np.argmin(dx), np.argmax(dx)
#         R = (dx[id_max] - dx[id_min]) / 2.0
#         C = X.mean(axis=0) + line_vec * ((dx[id_max] + dx[id_min]) / 2.0)
#         Xb = X[[id_min, id_max], :]
#         return SphereND(C, R), Xb
#
#     if D == 3 and is_coplanar(X, eps_angle):
#         # Project to 2D plane using first three points
#         # Choose plane spanned by first three points
#         p0, p1, p2 = X[:3]
#         v1, v2 = p1 - p0, p2 - p0
#         v1 /= np.linalg.norm(v1)
#         v2 -= v2.dot(v1) * v1  # orthogonalize
#         v2 /= np.linalg.norm(v2)
#         X2d = np.column_stack([(X - p0) @ v1, (X - p0) @ v2])
#         circle, Xb2d = exact_min_bounding_n_sphere(X2d, eps_angle)
#
#         R, C2d = circle.radius, circle.center
#         # Map back to 3D
#         C = p0 + v1 * C2d[0] + v2 * C2d[1]
#         Xb = p0 + np.outer(Xb2d[:, 0], v1) + np.outer(Xb2d[:, 1], v2)
#         return SphereND(C, R), Xb
#
#     # --- Reduce points via convex hull ------------------------------------
#     if len(X) > D + 1:
#         hull = ConvexHull(X)
#         X = X[hull.vertices]
#
#     # Remove duplicates
#     X = np.unique(X, axis=0)
#
#     if len(X) <= D + 1:
#         R, C = fit_circle_or_sphere_to_points(X, eps_angle)
#         return SphereND(C, R), X
#
#     # --- Randomly permute points for Welzl recursion ----------------------
#     X = np.random.permutation(X)
#
#     # Safe recursion
#     if len(X) < sys.getrecursionlimit():
#         R, C, Xb = B_min_enclosing_circle_or_sphere(X)
#         Xb = Xb[np.lexsort(Xb.T[::-1])]  # sort lexicographically
#         return SphereND(C, R), Xb
#
#     # --- Chunked fallback for large point clouds --------------------------
#     M = len(X)
#     dM = min(M // 4, 300)
#     X_splits = np.array_split(X, dM)
#     Xb = np.empty((0, D))
#     R, C = np.nan, np.full(D, np.nan)
#     for Xi in X_splits:
#         Xi = np.concatenate([Xi, Xb], axis=0) if len(Xb) > 0 else Xi
#         R, C, Xb = B_min_enclosing_circle_or_sphere(Xi)
#
#     Xb = Xb[np.lexsort(Xb.T[::-1])]
#     return SphereND(C, R), Xb
#
#
# # -------------------------------------------------------------------------
# # Exact minimum bounding ellipse (2D) or ellipsoid (3D) using Khachiyan's algorithm
# # -------------------------------------------------------------------------
# def min_bounding_ellipsoid_nd(
#     X: np.ndarray,
#     tol: float = 1e-9,
#     max_iter: int = 1000,
#     weight_thresh: float = 0.95,
#     return_hull_pts: bool = False,
#     return_weights: bool = False
# ):
#     """
#     Compute the minimum-volume enclosing ellipsoid of a point cloud in N dimensions
#     using Khachiyan's algorithm, handling degenerate (lower-rank) point clouds.
#
#     Parameters
#     ----------
#     X : (N_points, N_dims) ndarray
#         Input point cloud in N-dimensional space.
#     tol : float
#         Convergence tolerance for Khachiyan's algorithm.
#     max_iter : int
#         Maximum number of iterations.
#     weight_thresh : float
#         Fraction of cumulative sorted weights used to identify controlling points.
#         Set to 0 to skip boundary extraction.
#     return_hull_pts : bool
#         If True, also return the convex hull points used internally.
#     return_weights : bool
#         If True, also return the weight vector for all hull points.
#
#     Returns
#     -------
#     ellipsoid : EllipsoidND
#         Minimal enclosing ellipsoid (center and shape matrix A).
#     Xb : (M, N) ndarray, optional
#         Boundary points contributing to weight_thresh fraction.
#     X_hull : (K, N) ndarray, optional
#         Convex hull points used internally.
#     u : (K,) ndarray, optional
#         Final weights assigned to each convex hull point.
#     """
#     X = np.asarray(X)
#     N_points, D = X.shape
#     if N_points < 1:
#         raise ValueError("Input point cloud is empty.")
#
#     # --- Center points and check rank for degeneracy ---
#     X_centered = X - X[0]
#     rank = np.linalg.matrix_rank(X_centered, tol=tol)
#
#     if rank < D:
#         # Project points to intrinsic lower-dimensional subspace
#         U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
#         V_sub = Vt[:rank]             # basis of intrinsic subspace
#         X_sub = X_centered @ V_sub.T  # project onto subspace
#
#         # Recursively compute ellipsoid in lower-dimensional space
#         ellipsoid_sub, *extra = min_bounding_ellipsoid_nd(
#             X_sub, tol=tol, max_iter=max_iter,
#             weight_thresh=weight_thresh,
#             return_hull_pts=return_hull_pts,
#             return_weights=return_weights
#         )
#
#         # Map ellipsoid back to original N-D
#         c_nd = X[0] + V_sub.T @ ellipsoid_sub.center
#         A_nd = V_sub.T @ ellipsoid_sub.A @ V_sub
#
#         ellipsoid = EllipsoidND(c_nd, A_nd)
#
#         # Map extra outputs (boundary points / hull points) back to N-D
#         mapped_extra = []
#         for arr in extra:
#             if arr is None:  # not strictly required unless None is appended in future
#                 mapped_extra.append(None)
#             elif return_weights and arr.ndim == 1:
#                 # Weight vector, leave as-is
#                 mapped_extra.append(arr)
#             else:
#                 # Array of points, map back to N-D
#                 mapped_extra.append(X[0] + (V_sub.T @ arr.T).T)
#
#         return ellipsoid, *mapped_extra
#
#     # --- Full-rank case ---
#     # Reduce points via convex hull for numerical stability if more than D+1 points
#     if N_points > D + 1:
#         hull = ConvexHull(X)
#         X = X[hull.vertices]
#         N_points = len(X)
#
#     # Khachiyan initialization
#     Q = np.column_stack((X, np.ones(N_points)))  # augment for affine coordinates
#     u = np.ones(N_points) / N_points
#
#     for _ in range(max_iter):
#         V = Q.T @ np.diag(u) @ Q
#         M = np.einsum('ij,jk,ki->i', Q, np.linalg.inv(V), Q.T)
#         j = np.argmax(M)
#         max_M = M[j]
#         step_size = (max_M - D - 1) / ((D + 1) * (max_M - 1))
#         new_u = (1 - step_size) * u
#         new_u[j] += step_size
#         if np.linalg.norm(new_u - u) < tol:
#             u = new_u
#             break
#         u = new_u
#
#     # Compute ellipsoid center and shape matrix
#     c = X.T @ u
#     X_shift = X - c
#     A = np.linalg.inv(X_shift.T @ np.diag(u) @ X_shift) / D
#
#     return_object = [EllipsoidND(c, A)]
#
#     # Extract boundary points if requested
#     if weight_thresh > 0:
#         sorted_indices = np.argsort(u)[::-1]
#         sorted_weights = u[sorted_indices]
#         cumulative_sum_weights = np.cumsum(sorted_weights)
#         cutoff_idx = np.searchsorted(cumulative_sum_weights, weight_thresh)
#         controlling_point_indices = sorted_indices[:cutoff_idx + 1]
#         Xb = X[controlling_point_indices]
#         if Xb.shape[0] < D + 1:
#             Xb = X[sorted_indices[:D + 1]]
#         return_object.append(Xb)
#
#     if return_hull_pts:
#         return_object.append(X)
#     if return_weights:
#         return_object.append(u)
#
#     return tuple(return_object)
#
# def min_bounding_n_ellipsoid(X: np.ndarray, tol=1e-9, max_iter=1000, weight_thresh=0.95, return_hull_pts=False, return_weights=False):
#     """
#     Compute the minimum-volume enclosing ellipse (2D) or ellipsoid (3D)
#     of a point cloud using **Khachiyan's algorithm**.
#
#     The algorithm minimizes the volume of the ellipsoid satisfying:
#         (x - c)^T A (x - c) = 1
#     where `A` is a positive definite shape matrix and `c` is the center.
#
#     The function automatically determines whether to compute an ellipse or
#     ellipsoid based on the dimensionality of the input `X`.
#
#     Parameters
#     ----------
#     X : (N, D) ndarray
#         Input point cloud with `D = 2` (ellipse) or `D = 3` (ellipsoid).
#     tol : float, default=1e-9
#         Convergence tolerance for Khachiyan's algorithm.
#     max_iter : int, default=1000
#         Maximum number of iterations for convergence.
#     weight_thresh : float, default=0.95
#         Fraction (0–1) of the cumulative sum of sorted point weights `u`
#         used to identify the boundary (controlling) points of the ellipsoid.
#         Set to 0 to skip boundary point extraction.
#     return_hull_pts : bool, default=False
#         If True, also return the reduced convex hull points used internally.
#     return_weights : bool, default=False
#         If True, also return the weight vector `u` for all hull points.
#
#     Returns
#     -------
#     EllipsoidND
#         Instance of the `EllipsoidND` class containing:
#             - `center` : ndarray of shape (D,)
#             - `A` : shape matrix of shape (D, D)
#               such that (x - c)^T A (x - c) = 1
#
#     Xb : (M, D) ndarray, optional
#         Boundary or controlling points contributing cumulatively to
#         `weight_thresh` fraction of the total weight sum. Returned only if
#         `weight_thresh > 0`.
#
#     X : (K, D) ndarray, optional
#         The convex hull points used in the Khachiyan algorithm.
#         Returned only if `return_hull_pts=True`.
#
#     u : (K,) ndarray, optional
#         The final weights assigned to each convex hull point.
#         Returned only if `return_weights=True`.
#
#     Notes
#     -----
#     - The first return value is always an `EllipsoidND` (2D) or `Ellipsoid` (3D) object.
#     - Additional return values depend on the optional arguments provided.
#     - Internally, the algorithm operates on the convex hull of `X`
#       to improve numerical stability and efficiency.
#
#     """
#
#     X = np.asarray(X)
#     N, D = X.shape
#     if D not in (2, 3):
#         raise ValueError("Only 2D (ellipse) or 3D (ellipsoid) points are supported.")
#
#     # Reduce points via convex hull
#     if N > D + 1:
#         hull = ConvexHull(X)
#         X = X[hull.vertices]
#         N = len(X)
#
#     # ellipsoid equation
#     # (x−c).T * A(x−c) = 1
#     # Augment points for affine coordinates
#     Q = np.column_stack((X, np.ones(N)))
#     # initialise equal weights and ensure summation is equal to one
#     u = np.ones(N) / N
#
#     for _ in range(max_iter):
#         # Compute the weighted covariance of the points (in affine form), where each point contributes proportionally to its current weight ui
#         V = Q.T @ np.diag(u) @ Q
#         # weighted Mahalanobis distance test for the ellipsoid fit
#         # V^−1 ∼   [[   A   ,    -A*c   ],
#         #           [(-c.T*A), c.T*A*c−1]]
#         # Mi is a scalar that measures how “far” the point is from the current ellipsoid
#         M = np.einsum('ij,jk,ki->i', Q, np.linalg.inv(V), Q.T)
#         # Find the worst offender: the point with largest Mi (farthest outside)
#         j = np.argmax(M)
#         max_M = M[j]
#         # increase the weight uj slightly whilst reducing others
#         step_size = (max_M - D - 1) / ((D + 1) * (max_M - 1))
#         new_u = (1 - step_size) * u
#         new_u[j] += step_size
#         # tolerance check for early convergence
#         if np.linalg.norm(new_u - u) < tol:
#             u = new_u
#             break
#         u = new_u
#
#     # Compute center
#     c = X.T @ u
#
#     # Compute shape matrix
#     X_shift = X - c
#     A = np.linalg.inv(X_shift.T @ np.diag(u) @ X_shift) / D
#
#     return_object = [EllipsoidND(c, A)]
#
#     if weight_thresh > 0:
#
#         # sorted weights largest to smallest
#         sorted_indices  = np.argsort(u)[::-1]
#         sorted_weights = u[sorted_indices ]
#         cumulative_sum_weights = np.cumsum(sorted_weights)
#         # Find the index where cumulative sum exceeds weight_thresh
#         cutoff_idx = np.searchsorted(cumulative_sum_weights, weight_thresh)
#         # Indices of points that control the ellipsoid
#         controlling_point_indices = sorted_indices[:cutoff_idx + 1]
#
#         # Extract boundary points
#         Xb = X[controlling_point_indices]
#
#         # Ensure at least D+1 points
#         if Xb.shape[0] < D + 1:
#             Xb = X[sorted_indices[:D + 1]]
#
#         return_object.append(Xb)
#
#     if return_hull_pts:
#         return_object.append(X)
#     if return_weights:
#         return_object.append(u)
#
#     return tuple(return_object)
#
# def ellipsoid_residuals(X, A, c):
#     """
#     Compute residuals of points relative to the ellipsoid (x-c)^T A (x-c) = 1
#
#     Parameters
#     ----------
#     X : (N, d) ndarray
#         Point coordinates
#     A : (d, d) ndarray
#         Ellipsoid shape matrix
#     c : (d,) ndarray
#         Center of the ellipsoid
#
#     Returns
#     -------
#     r : (N,) ndarray
#         Residuals for each point
#     """
#     X_shift = X - c
#     r = np.einsum('ij,jk,ik->i', X_shift, A, X_shift) - 1
#     return r

# -------------------------------------------------------------------------
# Minimum Volume Enclosing Sphere (MVES) using Khachiyan-style iteration
# -------------------------------------------------------------------------
def _min_enclosing_sphere(X, tol=1e-7, max_iter=1000):
    """
    Compute the minimum enclosing sphere of a set of points using an
    iterative Khachiyan-like algorithm, enforcing A = 1/R^2 * I.

    Parameters
    ----------
    X : (N, d) ndarray
        Input points.
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    R : float
        Sphere radius.
    c : (d,) ndarray
        Sphere center.
    Xb : (k, d) ndarray
        Subset of points lying on the sphere boundary.
    """

    X = np.asarray(X)
    N, d = X.shape

    # Reduce to convex hull points for efficiency
    if N > d + 1:
        hull = ConvexHull(X)
        X = X[hull.vertices]
        N = X.shape[0]

    # Initialize center as mean, radius as max distance
    c = X.mean(axis=0)
    R = np.max(np.linalg.norm(X - c, axis=1))

    for _ in range(max_iter):
        # Compute distances from center
        diffs = X - c
        norms = np.linalg.norm(diffs, axis=1)
        max_idx = np.argmax(norms)
        R_new = norms[max_idx]

        if abs(R_new - R) < tol:
            R = R_new
            break

        # Move center slightly toward the farthest point
        c += (diffs[max_idx]) * (1 - R / R_new) / 2
        R = R_new

    # Identify boundary points (within tolerance)
    Xb = X[np.abs(np.linalg.norm(X - c, axis=1) - R) < tol * 10]

    return R, c, Xb


__version__ = "0.1.0"

__all__ = [
    "EllipsoidND",
    "SphereND",
    "intrinsic_dimension",
    "is_collinear",
    "is_coplanar",
]


