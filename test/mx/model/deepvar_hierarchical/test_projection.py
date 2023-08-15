from typing import Optional

import numpy as np
import pytest

from gluonts.mx.model.deepvar_hierarchical._estimator import (
    constraint_mat,
    projection_mat,
)


TOL = 1e-12

S = np.array(
    [
        [1, 1, 1, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ],
)

num_bottom_ts = S.shape[1]
A = constraint_mat(S)


def null_space_projection_mat(
    A: np.ndarray,
    D: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Computes the projection matrix for projecting onto the null space of A.

    Parameters
    ----------
    A
        The constraint matrix A in the equation: Ay = 0 (y being the
        values/forecasts of all time series in the hierarchy).
    D
        Symmetric positive definite matrix (typically a diagonal matrix).
        Optional.
        If provided then the distance between the reconciled and unreconciled
        forecasts is calculated based on the norm induced by D. Useful for
        weighing the distances differently for each level of the hierarchy.
        By default Euclidean distance is used.

    Returns
    -------
    Numpy ND array
        Projection matrix, shape (total_num_time_series, total_num_time_series)
    """
    num_ts = A.shape[1]
    if D is None:
        return np.eye(num_ts) - A.T @ np.linalg.pinv(A @ A.T) @ A
    else:
        assert np.all(D == D.T), "`D` must be symmetric."
        assert np.all(
            np.linalg.eigvals(D) > 0
        ), "`D` must be positive definite."

        D_inv = np.linalg.inv(D)
        return (
            np.eye(num_ts) - D_inv @ A.T @ np.linalg.pinv(A @ D_inv @ A.T) @ A
        )


@pytest.mark.parametrize(
    "D",
    [
        None,
        # Root gets the maximum weight and the two aggregated levels get
        # more weight than the leaf level.
        np.diag([4, 2, 2, 1, 1, 1, 1]),
        # Random diagonal matrix
        np.diag(np.random.rand(S.shape[0])),
        # Random positive definite matrix
        np.diag(np.random.rand(S.shape[0]))
        + np.dot(
            np.array([[4, 2, 2, 1, 1, 1, 1]]).T,
            np.array([[4, 2, 2, 1, 1, 1, 1]]),
        ),
    ],
)
def test_projection_mat(D):
    p1 = null_space_projection_mat(A=A, D=D)
    p2 = projection_mat(S=S, D=D)
    assert (np.abs(p1 - p2)).sum() < TOL
