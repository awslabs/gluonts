import warnings
from typing import Tuple

import numpy as np
import torch
from torch import matmul


def regularize_matrix(mat, regularization, relative_to_diag_mean=True):
    if regularization > 0:
        assert len(set(mat.shape[-2:])) == 1, "matrix must be square"
        if relative_to_diag_mean:
            diag_mean = batch_diag(mat).mean(dim=-1, keepdim=True)
            diag_regularization = batch_diag_matrix(
                torch.ones_like(mat[..., 0]) * diag_mean * regularization
            )
        else:
            diag_regularization = batch_diag_matrix(
                torch.ones_like(mat[..., 0]) * regularization
            )
        regularized_mat = mat + diag_regularization
    else:
        regularized_mat = mat
    return regularized_mat


def cholesky(
    mat,
    upper=False,
    out=None,
    regularization=0.0,
    initial_try_regularization=1e-6,
    max_regularization=1e2,
    step_factor=10,
    warning=True,
):
    if regularization > 0 and warning:
        warnings.warn(f"regularizing matrix with value: {regularization}")
        regularized_mat = regularize_matrix(
            mat=mat, regularization=regularization, relative_to_diag_mean=True
        )
    else:
        regularized_mat = mat
    try:
        # TODO: there is currently a magma bug that occurs for certain batch sizes.
        #  github issue: https://github.com/pytorch/pytorch/issues/26996
        #  see also https://discuss.pytorch.org/t/cuda-illegal-memory-access-when-using-batched-torch-cholesky/51624/8
        #  for the temporary hack used below
        # device = regularized_mat.device
        # return torch.cholesky(regularized_mat.to("cpu"), upper=upper, out=out).to(device)
        # TODO: bug still exists, but only for certain batch sizes. This slows down too much
        return torch.cholesky(regularized_mat, upper=upper, out=out)
    except RuntimeError:
        if torch.isnan(mat).any():
            raise ValueError("NaN detected in input mat.")
        else:
            regularization = max(
                regularization * step_factor, initial_try_regularization
            )
            if regularization > max_regularization:
                raise ValueError(
                    f"attempted regularization greater than "
                    f"max_regularization: {max_regularization}."
                )
            else:
                return cholesky(
                    mat=mat,
                    upper=upper,
                    out=out,
                    regularization=regularization,
                )


def cov_and_chol_from_invcholesky_param(Linv_tril, Linv_logdiag):
    L = torch.inverse(
        torch.tril(Linv_tril, -1) + batch_diag_matrix(torch.exp(Linv_logdiag))
    )
    return matmul(L, L.transpose(-1, -2)), L


def cov_from_invcholesky_param(Linv_tril, Linv_logdiag):
    return cov_and_chol_from_invcholesky_param(
        Linv_tril=Linv_tril, Linv_logdiag=Linv_logdiag
    )[0]


def inv_from_invcholesky_param(Linv_tril, Linv_logdiag):
    Linv = torch.tril(Linv_tril, -1) + batch_diag_matrix(
        torch.exp(Linv_logdiag)
    )
    return matmul(Linv.transpose(-1, -2), Linv)


def batch_cholesky_inverse(Lmat):
    """
    performs torch.cholesky_inverse for matrices with leading
    match dimensions.
    Note that torch can do batched torch.cholesky but not torch.cholesky_inverse
    """
    Lmatinv = torch.inverse(Lmat)
    return matmul(Lmatinv.transpose(-1, -2), Lmatinv)


def batch_diag(mat):
    return mat[..., torch.arange(mat.shape[-2]), torch.arange(mat.shape[-1])]


def batch_diag_matrix(diagonal):
    N = diagonal.shape[-1]
    shape = diagonal.shape[:-1] + (N, N)
    device, dtype = diagonal.device, diagonal.dtype
    result = torch.zeros(shape, dtype=dtype, device=device)
    indices = torch.arange(result.numel(), device=device).reshape(shape)
    indices = indices.diagonal(dim1=-2, dim2=-1)
    result.view(-1)[indices] = diagonal
    return result


def symmetrize(mat):
    return 0.5 * (mat + mat.transpose(-1, -2))


def matvec(mat, vec):
    if vec.ndim == 1:
        return torch.matmul(mat, vec)
    elif mat.ndim == 2:
        return torch.matmul(vec, mat.transpose(1, 0))
    else:
        return torch.matmul(mat, vec[..., None])[..., 0]


def get_numerical_precision(x):
    if isinstance(x, float):
        eps = np.finfo(np.array(x).dtype).eps
    elif isinstance(x, (torch.Tensor,)):
        eps = torch.finfo(x.dtype).eps
    elif isinstance(x, (np.ndarray, np.generic)):
        eps = np.finfo(x.dtype).eps
    else:
        raise ValueError("unknown dtype of {}".format(x))
    return eps


def stable_log(x, eps=None):
    eps = get_numerical_precision(x) if eps is None else eps
    return torch.log(x + eps)


# def kron(A, B):
#     res_vec = torch.matmul(A[:, None, :, None], B[None, :, None, :])
#     return res_vec.reshape([A.shape[0] * B.shape[0], A.shape[1] * B.shape[1]])


def kron(A, B):
    """
    Kronecker Product.
    Works with batch dimemsion(s) - requires both A and B have same batch dims.
    """
    A_shp, B_shp = A.shape, B.shape
    assert A_shp[:-2] == B_shp[:-2]
    kron_block = torch.matmul(
        A[..., :, None, :, None], B[..., None, :, None, :]
    )
    kron = kron_block.reshape(
        A_shp[:-2] + (A_shp[-2] * B_shp[-2], A_shp[-1] * B_shp[-1],)
    )
    return kron


def vectorize_normal_dist_params(m, V):
    # Take lower-triangular incl. diagonal from covariance matrix of state.
    indices = torch.tril_indices(row=V.shape[-2], col=V.shape[-1], offset=0)
    V_vectorised = V[..., indices[0], indices[1]]
    return torch.cat((m, V_vectorised), dim=-1)


def logmeanexp(self: torch.Tensor, dim: int, keepdim: bool = False):
    return torch.logsumexp(self, dim=dim, keepdim=keepdim) - np.log(
        self.shape[dim]
    )


def make_block_diagonal(mats: Tuple[torch.Tensor]) -> torch.Tensor:
    if not isinstance(mats, (list, tuple)):
        raise Exception("provide list or tuple")
    elif len(mats) > 1:
        off_diag_upper_right = torch.zeros(
            mats[-2].shape[:-1] + (mats[-1].shape[-1],),
            dtype=mats[-1].dtype,
            device=mats[-1].device,
        )
        off_diag_lower_left = torch.zeros(
            mats[-1].shape[:-1] + (mats[-2].shape[-1],),
            dtype=mats[-1].dtype,
            device=mats[-1].device,
        )
        block_diagonal = torch.cat(
            [
                torch.cat([mats[-2], off_diag_upper_right], dim=-1),
                torch.cat([off_diag_lower_left, mats[-1]], dim=-1),
            ],
            dim=-2,
        )
        return make_block_diagonal(tuple(mats[:-2]) + (block_diagonal,))
    else:  # final recursion
        return mats[0]
