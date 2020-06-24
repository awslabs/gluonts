import torch
from torch import matmul
from torch_extensions.ops import (
    batch_cholesky_inverse,
    symmetrize,
    matvec,
    cholesky,
)


def filter_forward_prediction_step(m, V, R, A, b=None):
    """ Single prediction step in forward filtering cycle
    (prediction --> measurement) """
    mp = matvec(A, m) if A is not None else m
    if b is not None:
        mp += b
    Vp = symmetrize(
        (matmul(A, matmul(V, A.transpose(-1, -2))) if A is not None else V) + R
    )
    return mp, Vp


def filter_forward_predictive_distribution(m, V, Q, C, d=None):
    """ The predictive distirbution p(yt | y_{1:t-1}),
    obtained by marginalising the state prediction distribution in the measurement model. """
    Vpy = symmetrize(matmul(C, matmul(V, C.transpose(-1, -2))) + Q)
    mpy = matvec(C, m)
    if d is not None:
        mpy += d
    return mpy, Vpy


def filter_forward_measurement_step(
    y, m, V, Q, C, d=None, return_loss_components=False
):
    """ Single measurement/update step in forward filtering cycle
    (prediction --> measurement) """
    mpy, Vpy = filter_forward_predictive_distribution(m=m, V=V, Q=Q, C=C, d=d)
    LVpyinv = torch.inverse(cholesky(Vpy))
    S = matmul(
        LVpyinv, matmul(C, V)
    )  # CV is cov_T. cov VC.T could be output of predictive dist.
    dobs = y - mpy
    dobs_norm = matvec(LVpyinv, dobs)
    mt = m + matvec(S.transpose(-1, -2), dobs_norm)
    Vt = symmetrize(V - matmul(S.transpose(-1, -2), S))
    if return_loss_components:  # I hate to do that! but it is convenient...
        return mt, Vt, dobs_norm, LVpyinv
    else:
        return mt, Vt


def smooth_backward_step(m_sm, V_sm, m_fw, V_fw, A, b, R):
    # filter one-step predictive variance
    P = (
        matmul(A, matmul(V_fw, A.transpose(-1, -2))) if A is not None else V_fw
    ) + R
    Pinv = batch_cholesky_inverse(cholesky(P))
    # m and V share J when in the conditioning operation (joint to posterior bw)
    J = matmul(V_fw, matmul(A.transpose(-1, -2), Pinv))

    m_sm_t = m_fw + matvec(
        J, m_sm - (matvec(A, m_fw) if A is not None else m_fw) - b
    )
    V_sm_t = symmetrize(
        V_fw + matmul(J, matmul(V_sm - P, J.transpose(-1, -2)))
    )
    Cov_sm_t = matmul(J, V_sm)
    return m_sm_t, V_sm_t, Cov_sm_t
