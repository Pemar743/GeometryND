import sys

from scipy.spatial import ConvexHull
from scipy.optimize import least_squares
import numpy as np
import math
from abc import ABC, abstractmethod


class GeometryND(ABC):
    """Abstract base class for n-dimensional geometric shapes."""
    
    @abstractmethod
    def __repr__(self):
        """Return a string representation of the shape."""
        pass

    @property
    @abstractmethod
    def n(self):
        """Ambient dimension of the shape."""
        pass
    
    @property
    @abstractmethod
    def volume(self):
        """Compute the n-dimensional volume of the shape."""
        pass

    @abstractmethod
    def get_residuals(self, X):
        """Compute residuals of points relative to the shape."""
        pass

    @abstractmethod
    def minimum_enclosing(self, X, **kwargs):
        """Compute the minimum enclosing shape for a set of points."""
        pass

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
        rank = GeometryND.intrinsic_dimension(points, tol=tol, return_centered_pts=False)
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
    
    # -------------------------------------------------------------------------
    # Helper functions for geometric checks
    # -------------------------------------------------------------------------
    @staticmethod
    def is_in_k_subspace( points, k, tol=1e-6):
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
        return GeometryND.intrinsic_dimension(points, tol=tol) <= k

    @staticmethod
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

    @staticmethod
    def is_collinear(points, tol=1e-6):
        """Check if the first three points are nearly collinear in N-D."""
        return GeometryND.is_in_k_subspace(points, k=1, tol=tol)

    @staticmethod
    def is_coplanar(points, tol=1e-6):
        """Check if the first four points are nearly coplanar in N-D."""
        return GeometryND.is_in_k_subspace(points, k=2, tol=tol)

class EllipsoidND(GeometryND):
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
            # X_sub, basis = cls.project_to_subspace(centered, intrinsic_dimension=rank)

        X_sub, basis, offset, is_in_subspace = GeometryND.project_to_subspace(X, tol=tol)
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
        # ConvexHull is only valid for dimension >= 2.
        if D >= 2 and N_points > D + 1:
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
        rank = GeometryND.intrinsic_dimension(points, tol=tol)

        if rank < M - 1:
            # Degenerate: no unique sphere possible
            return np.inf, np.full(D, np.nan)

        # ------------------------------------------------------------------------
        # Project points into intrinsic lower-dimensional subspace if rank < D
        # ------------------------------------------------------------------------
        if rank < D:
            X_sub, basis, offset, _ = GeometryND.project_to_subspace(points, tol=tol)
        # if is_in_subspace:
            # Recursively fit in lower-dimensional space
            r_sub, c_sub = SphereND.fit_circumsphere_nd(X_sub, tol=tol)
            # Lift center back to original space
            center_nd = GeometryND.lift_from_subspace(c_sub, basis, offset)
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
    def _lstsq_fit(cls, X):
        """
        Fit a sphere to a set of points using least squares.
        This is a helper method for best_fit and RANSAC.
        """
        X = np.asarray(X)
        N, d = X.shape

        # construct generalised matrix A: columns are 2*coords, final column is 1
        A = np.zeros((N, d + 1))
        A[:, :d] = 2 * X
        A[:, d] = 1

        # construct vector b: sum of squared coords for each point
        b = np.sum(X ** 2, axis=1)

        # Solve least squares: A @ [c; r^2 - |c|^2] = b
        sol, residuals, rank_ls, s = np.linalg.lstsq(A, b, rcond=None)
        center = sol[:d]
        radius_sq = sol[d] + np.sum(center ** 2)
        radius = np.sqrt(radius_sq)

        return center, radius

    @classmethod
    def _non_linear_lstsq_refine(cls, X, center, radius):
        """
        Refine the sphere fit using non-linear least squares.
        """
        # Optional: refine using non-linear least squares (e.g., Levenberg-Marquardt)
        
        d = X.shape[1]

        def residuals_func(params):
            c = params[:d]
            r = params[d]
            return np.linalg.norm(X - c, axis=1) - r

        initial_guess = np.hstack([center, radius])
        result = least_squares(residuals_func, initial_guess)
        center = result.x[:d]
        radius = result.x[d]

        return center, radius
    
    @classmethod
    def _ransac(cls, X, max_iter=100, tol=1e-3):
        """
        Internal method to perform RANSAC fitting for the sphere.
        """
        X = np.asarray(X)
        N, d = X.shape
        best_inliers = []
        best_model = None

        # Minimum points required to fit a sphere is n + 1 (for D dimensions)
        min_pts = d + 1

        if N < min_pts:
            raise ValueError(f"Need at least {min_pts} points to fit {d}-D sphere.")

        for _ in range(max_iter):
            # Random subset
            idx = np.random.choice(N, min_pts, replace=False)
            subset = X[idx]

            try:
                center, radius = cls._lstsq_fit(subset)
            except ValueError:
                continue

            # Compute residuals
            sphere_model = cls(center=center, radius=radius)
            res = sphere_model.get_residuals(X)
            inliers = np.where(np.abs(res) <= tol)[0]

            if len(inliers) > len(best_inliers):
                best_inliers = X[inliers]
                best_model = sphere_model
        if best_model is None:
            raise RuntimeError("RANSAC failed to find a valid sphere.")
        return best_inliers, best_model.center, best_model.radius

    @classmethod
    def best_fit(cls, X: np.ndarray, tol: float = 1e-9, ransac_iter: int = 0, ransac_tol: float = 1e-3, non_linear: bool = False):
        """
        Fit an n-dimensional sphere to a set of points using least squares,
        automatically handling degenerate (lower-rank) point clouds.

        Parameters
        ----------
        X : (N, n) ndarray
            Input point cloud.
        tol : float
            Tolerance for detecting rank deficiency / degeneracy.
        ransac_iter : int
            Number of iterations for RANSAC. If <= 0, RANSAC is not used.
        ransac_tol : float
            Tolerance for RANSAC inlier threshold.
        non_linear : bool
            If True, refine the fit using non-linear least squares after the initial linear fit.

        Returns
        -------
        SphereND
            Sphere fitted to the point cloud.
        """
        X = np.asarray(X)
        N, d = X.shape

        tol = abs(tol)
        ransac_tol = abs(ransac_tol)

        # --- Check for points being wholly in lower subspace ---
        X_proj, basis, offset, is_in_subspace = cls.project_to_subspace(X, tol=tol)
        if is_in_subspace:
            # recursively fit in lower-dim space
            sphere_sub = cls.best_fit(X_proj, tol=tol,
                                      ransac_iter=ransac_iter,
                                      ransac_tol=ransac_tol,
                                      non_linear=non_linear)
            # Map center back to original N-D
            center_nd = cls.lift_from_subspace(sphere_sub.center, basis, offset)
            return cls(center_nd, sphere_sub.radius)

        # --- Full-rank case ---
        # Fit circumsphere to points
        if N > d + 1:
            if ransac_iter > 0:  # Use RANSAC to handle outliers
                X, center, radius = cls._ransac(X, max_iter=ransac_iter, tol=ransac_tol)
            
            # get linear least squares fit
            center, radius = cls._lstsq_fit(X)

            if non_linear:  # Refine with non-linear least squares
                center, radius = cls._non_linear_lstsq_refine(X, center, radius)
        else:
            radius, center = cls.fit_circumsphere_nd(X, tol=tol)
            if np.isinf(radius) or np.isnan(radius):
                raise ValueError("Degenerate point configuration; cannot fit a unique sphere.")

        return cls(center=center, radius=radius)

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
        X_sub, basis, offset, is_in_subspace = GeometryND.project_to_subspace(X)
        if is_in_subspace:
            if X_sub.shape[1] > 1 and len(X_sub) > X_sub.shape[1] + 1:
                hull = ConvexHull(X_sub)
                X_sub = X_sub[hull.vertices]
            X_sub = np.unique(X_sub, axis=0)
            # --- Randomly permute points for Welzl recursion ----------------------
            X_sub = np.random.permutation(X_sub)
            r_sub, c_sub, boundary_sub = _welzl_exact_minimum_enclosing_nd_sphere(X_sub, eps=eps)
            # center_nd, A_nd = cls.lift_from_subspace(c_sub, r_sub, basis, offset)
            # Map center and boundary points back to original space
            center_nd = GeometryND.lift_from_subspace(c_sub, basis, offset)
            Xb_nd = GeometryND.lift_from_subspace(boundary_sub, basis, offset)

            return _return_(center=center_nd, radius=r_sub, Xb=Xb_nd)
    
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
    "GeometryND"
    "EllipsoidND",
    "SphereND",
]


