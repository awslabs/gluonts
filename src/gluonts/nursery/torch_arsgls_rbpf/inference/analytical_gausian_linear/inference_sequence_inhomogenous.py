from typing import Optional
import torch
from torch import matmul
from torch_extensions.ops import (
    batch_cholesky_inverse,
    batch_diag,
    kron,
    symmetrize,
    matvec,
    make_block_diagonal,
    cov_from_invcholesky_param,
    inv_from_invcholesky_param,
    cholesky,
)
from inference.analytical_gausian_linear.inference_step import (
    filter_forward_prediction_step,
    filter_forward_measurement_step,
    smooth_backward_step,
)
from utils.utils import (
    create_zeros_state_vec,
    create_zeros_state_mat,
    add_sample_dims_to_initial_state,
    LOG_2PI,
    TensorDims,
)


def filter_forward(
    dims: TensorDims,
    A: torch.Tensor,
    B: Optional[torch.Tensor],
    C: torch.Tensor,
    D: Optional[torch.Tensor],
    LQinv_tril: torch.Tensor,
    LQinv_logdiag: torch.Tensor,
    LRinv_tril: torch.Tensor,
    LRinv_logdiag: torch.Tensor,
    LV0inv_tril: torch.Tensor,
    LV0inv_logdiag: torch.Tensor,
    m0: torch.Tensor,
    y: torch.Tensor,
    u_state: Optional[torch.Tensor] = None,
    u_obs: Optional[torch.Tensor] = None,
):
    # TODO: assumes B != None, D != None. u_state and u_obs can be None.
    #  Better to work with vectors b, d. matmul Bu should be done outside!
    m_fw = create_zeros_state_vec(dims=dims, device=A.device, dtype=A.dtype)
    V_fw = create_zeros_state_mat(dims=dims, device=A.device, dtype=A.dtype)
    for t in range(0, dims.timesteps):
        if t == 0:
            V0 = cov_from_invcholesky_param(LV0inv_tril, LV0inv_logdiag)
            mp, Vp = add_sample_dims_to_initial_state(m0=m0, V0=V0, dims=dims)
        else:
            R_tm1 = cov_from_invcholesky_param(
                LRinv_tril[t - 1], LRinv_logdiag[t - 1]
            )
            b_tm1 = (
                matvec(B[t - 1], u_state[t - 1]) if u_state is not None else 0
            )
            mp, Vp = filter_forward_prediction_step(
                m=m_fw[t - 1], V=V_fw[t - 1], R=R_tm1, A=A[t - 1], b=b_tm1
            )
        Q_t = cov_from_invcholesky_param(LQinv_tril[t], LQinv_logdiag[t])
        d_t = matvec(D[t], u_obs[t]) if u_obs is not None else 0
        m_fw[t], V_fw[t] = filter_forward_measurement_step(
            y=y[t], m=mp, V=Vp, Q=Q_t, C=C[t], d=d_t
        )
    return torch.stack(m_fw, dim=0), torch.stack(V_fw, dim=0)


def smooth_forward_backward(
    dims: TensorDims,
    A: torch.Tensor,
    B: Optional[torch.Tensor],
    C: torch.Tensor,
    D: Optional[torch.Tensor],
    LQinv_tril: torch.Tensor,
    LQinv_logdiag: torch.Tensor,
    LRinv_tril: torch.Tensor,
    LRinv_logdiag: torch.Tensor,
    LV0inv_tril: torch.Tensor,
    LV0inv_logdiag: torch.Tensor,
    m0: torch.Tensor,
    y: torch.Tensor,
    u_state: Optional[torch.Tensor] = None,
    u_obs: Optional[torch.Tensor] = None,
):
    m_sm = create_zeros_state_vec(dims=dims, device=A.device, dtype=A.dtype)
    V_sm = create_zeros_state_mat(dims=dims, device=A.device, dtype=A.dtype)
    Cov_sm = create_zeros_state_mat(dims=dims, device=A.device, dtype=A.dtype)
    m_fw, V_fw = filter_forward(
        dims=dims,
        A=A,
        B=B,
        C=C,
        D=D,
        LQinv_tril=LQinv_tril,
        LQinv_logdiag=LQinv_logdiag,
        LRinv_tril=LRinv_tril,
        LRinv_logdiag=LRinv_logdiag,
        LV0inv_tril=LV0inv_tril,
        LV0inv_logdiag=LV0inv_logdiag,
        m0=m0,
        y=y,
        u_state=u_state,
        u_obs=u_obs,
    )
    m_sm[-1], V_sm[-1] = m_fw[-1], V_fw[-1]
    for t in reversed(range(0, dims.timesteps - 1)):
        Rt = cov_from_invcholesky_param(LRinv_tril[t], LRinv_logdiag[t])
        bt = matvec(B[t], u_state[t]) if u_state is not None else 0
        m_sm[t], V_sm[t], Cov_sm[t] = smooth_backward_step(
            m_sm=m_sm[t + 1],
            V_sm=V_sm[t + 1],
            m_fw=m_fw[t],
            V_fw=V_fw[t],
            A=A[t],
            R=Rt,
            b=bt,
        )
    return (
        torch.stack(m_sm, dim=0),
        torch.stack(V_sm, dim=0),
        torch.stack(Cov_sm, dim=0),
    )


def smooth_global(
    dims: TensorDims,
    A: torch.Tensor,
    B: Optional[torch.Tensor],
    C: torch.Tensor,
    D: Optional[torch.Tensor],
    LQinv_tril: torch.Tensor,
    LQinv_logdiag: torch.Tensor,
    LRinv_tril: torch.Tensor,
    LRinv_logdiag: torch.Tensor,
    LV0inv_tril: torch.Tensor,
    LV0inv_logdiag: torch.Tensor,
    m0: torch.Tensor,
    y: torch.Tensor,
    u_state: Optional[torch.Tensor] = None,
    u_obs: Optional[torch.Tensor] = None,
):
    """ compute posterior by direct inversion of unrolled model """
    # This implementation works only if all mats have time and batch dimension.
    #  Otherwise does not broadcast correctly.
    assert A.ndim == 4
    assert B.ndim == 4
    assert C.ndim == 4
    assert D.ndim == 4
    assert LQinv_tril.ndim == 4
    assert LQinv_logdiag.ndim == 3
    assert LRinv_tril.ndim == 4
    assert LRinv_logdiag.ndim == 3

    # raise NotImplementedError("Not yet implemented")
    device, dtype = A.device, A.dtype

    R = cov_from_invcholesky_param(LRinv_tril, LRinv_logdiag)
    Q = cov_from_invcholesky_param(LQinv_tril, LQinv_logdiag)
    V0 = cov_from_invcholesky_param(LV0inv_tril, LV0inv_logdiag)

    Q_field = torch.zeros(
        (dims.batch, dims.timesteps * dims.state, dims.timesteps * dims.state),
        device=device,
        dtype=dtype,
    )
    h_field = torch.zeros(
        (dims.batch, dims.timesteps * dims.state), device=device, dtype=dtype
    )

    b = matvec(B, u_state[:-1]) if u_state is not None else 0
    d = matvec(D, u_obs) if u_obs is not None else 0

    Rinv = symmetrize(batch_cholesky_inverse(cholesky(R)))
    Qinv = symmetrize(batch_cholesky_inverse(cholesky(Q)))
    V0inv = symmetrize(batch_cholesky_inverse(cholesky(V0)))

    CtQinvymd = matvec(matmul(C.transpose(-1, -2), Qinv), y - d)
    h_obs = CtQinvymd.transpose(1, 0).reshape(
        (dims.batch, dims.timesteps * dims.state,)
    )

    CtQinvC = matmul(C.transpose(-1, -2), matmul(Qinv, C))
    assert len(CtQinvC) == dims.timesteps
    Q_obs = make_block_diagonal(mats=tuple(mat_t for mat_t in CtQinvC))

    AtRinvA = matmul(A.transpose(-1, -2), matmul(Rinv, A))
    RinvA = matmul(Rinv, A)

    h_field[:, : dims.state] = matmul(V0inv, m0).repeat(
        (dims.batch,) + (1,) * (h_field.ndim - 1)
    )
    Q_field[:, : dims.state, : dims.state] += V0inv.repeat(
        (dims.batch,) + (1,) * (Q_field.ndim - 1)
    )
    for t in range(dims.timesteps - 1):
        id = t * dims.state
        h_field[:, id : id + dims.state] += -matvec(
            RinvA[t].transpose(-1, -2), b[t]
        )
        h_field[:, id + dims.state : id + 2 * dims.state] += matvec(
            Rinv[t], b[t]
        )
        Q_field[:, id : id + dims.state, id : id + dims.state] += AtRinvA[t]
        Q_field[
            :, id : id + dims.state, id + dims.state : id + 2 * dims.state
        ] += -RinvA[t].transpose(-1, -2)
        Q_field[
            :, id + dims.state : id + 2 * dims.state, id : id + dims.state
        ] += -RinvA[t]
        Q_field[
            :,
            id + dims.state : id + 2 * dims.state,
            id + dims.state : id + 2 * dims.state,
        ] += Rinv[t]

    L_all_inv = torch.inverse(cholesky(Q_field + Q_obs))
    V_all = matmul(L_all_inv.transpose(-1, -2), L_all_inv)
    m_all = matvec(V_all, h_obs + h_field)

    # Pytorch has no Fortran style reading of indices.
    m = m_all.reshape((dims.batch, dims.timesteps, dims.state)).transpose(0, 1)
    V, Cov = [], []
    for t in range(0, dims.timesteps):
        id = t * dims.state
        V.append(V_all[:, id : id + dims.state, id : id + dims.state])
        if t < (dims.timesteps - 1):
            Cov.append(
                V_all[
                    :,
                    id : id + dims.state,
                    id + dims.state : id + 2 * dims.state,
                ]
            )
        else:
            Cov.append(
                torch.zeros(
                    (dims.batch, dims.state, dims.state),
                    device=device,
                    dtype=dtype,
                )
            )
    V = torch.stack(V, dim=0)
    Cov = torch.stack(Cov, dim=0)

    return m, V, Cov


def sample(
    dims: TensorDims,
    A: torch.Tensor,
    B: Optional[torch.Tensor],
    C: torch.Tensor,
    D: Optional[torch.Tensor],
    LQinv_tril: torch.Tensor,
    LQinv_logdiag: torch.Tensor,
    LRinv_tril: torch.Tensor,
    LRinv_logdiag: torch.Tensor,
    LV0inv_tril: torch.Tensor,
    LV0inv_logdiag: torch.Tensor,
    m0: torch.Tensor,
    u_state: Optional[torch.Tensor] = None,
    u_obs: Optional[torch.Tensor] = None,
):
    # generate noise
    wz = torch.randn(dims.timesteps, dims.batch, dims.state)
    wy = torch.randn(dims.timesteps, dims.batch, dims.target)

    # Initial step.
    # Note: We cannot use in-place operations here because we must backprop through y.
    LV0 = torch.inverse(
        torch.tril(LV0inv_tril, -1) + torch.diag(torch.exp(LV0inv_logdiag))
    )
    LQ_0 = torch.inverse(
        torch.tril(LQinv_tril[0], -1) + torch.diag(torch.exp(LQinv_logdiag[0]))
    )
    d_0 = matvec(D[0], u_obs[0]) if u_obs is not None else 0
    x = [m0 + matvec(LV0, wz[0])] + [None] * (dims.timesteps - 1)
    y = [matvec(C[0], x[0]) + d_0 + matvec(LQ_0, wy[0])] + [None] * (
        dims.timesteps - 1
    )
    for t in range(1, dims.timesteps):
        LR_tm1 = torch.inverse(
            torch.tril(LRinv_tril[t - 1], -1)
            + torch.diag(torch.exp(LRinv_logdiag[t - 1]))
        )
        LQ_t = torch.inverse(
            torch.tril(LQinv_tril[t], -1)
            + torch.diag(torch.exp(LQinv_logdiag[t]))
        )
        b_tm1 = matvec(B[t - 1], u_state[t - 1]) if u_state is not None else 0
        d_t = matvec(D[t], u_obs[t]) if u_obs is not None else 0

        x[t] = matvec(A[t - 1], x[t - 1]) + b_tm1 + matvec(LR_tm1, wz[t])
        y[t] = matvec(C[t], x[t]) + d_t + matvec(LQ_t, wy[t])

    return torch.stack(x, dim=0), torch.stack(y, dim=0)


def loss_forward(
    dims: TensorDims,
    A: torch.Tensor,
    B: Optional[torch.Tensor],
    C: torch.Tensor,
    D: Optional[torch.Tensor],
    LQinv_tril: torch.Tensor,
    LQinv_logdiag: torch.Tensor,
    LRinv_tril: torch.Tensor,
    LRinv_logdiag: torch.Tensor,
    LV0inv_tril: torch.Tensor,
    LV0inv_logdiag: torch.Tensor,
    m0: torch.Tensor,
    y: torch.Tensor,
    u_state: Optional[torch.Tensor] = None,
    u_obs: Optional[torch.Tensor] = None,
):
    """
    Computes the sample-wise (e.g. (particle, batch)) forward / filter loss.
    If particle and batch dims are used,
    the ordering is TPBF (time, particle, batch, feature).

    Note: it would be better numerically (amount of computation and precision)
    to sum/avg over these dimensions in all loss terms.
    However, we must compute it sample-wise for many algorithms,
    i.e. Rao-Blackwellised Particle Filters.
    """
    device, dtype = A.device, A.dtype

    V0 = cov_from_invcholesky_param(LV0inv_tril, LV0inv_logdiag)
    m0, V0 = add_sample_dims_to_initial_state(m0=m0, V0=V0, dims=dims)

    # Note: We can not use (more readable) in-place operations due to backprop problems.
    m_fw = [None] * dims.timesteps
    V_fw = [None] * dims.timesteps
    dim_particle = (
        (dims.particle,)
        if dims.particle is not None and dims.particle != 0
        else tuple()
    )
    loss = torch.zeros(
        dim_particle + (dims.batch,), device=device, dtype=dtype
    )
    for t in range(0, dims.timesteps):
        (mp, Vp) = (
            filter_forward_prediction_step(
                m=m_fw[t - 1],
                V=V_fw[t - 1],
                R=cov_from_invcholesky_param(
                    LRinv_tril[t - 1], LRinv_logdiag[t - 1]
                ),
                A=A[t - 1],
                b=matvec(B[t - 1], u_state[t - 1])
                if u_state is not None
                else 0,
            )
            if t > 0
            else (m0, V0)
        )
        m_fw[t], V_fw[t], dobs_norm, LVpyinv = filter_forward_measurement_step(
            y=y[t],
            m=mp,
            V=Vp,
            Q=cov_from_invcholesky_param(LQinv_tril[t], LQinv_logdiag[t]),
            C=C[t],
            d=matvec(D[t], u_obs[t]) if u_obs is not None else 0,
            return_loss_components=True,
        )
        loss += 0.5 * (
                torch.sum(dobs_norm ** 2, dim=-1)
                - 2 * torch.sum(torch.log(batch_diag(LVpyinv)), dim=(-1,))
                + dims.target * LOG_2PI
        )
    return loss


def loss_em(
    dims: TensorDims,
    A: torch.Tensor,
    B: Optional[torch.Tensor],
    C: torch.Tensor,
    D: Optional[torch.Tensor],
    LQinv_tril: torch.Tensor,
    LQinv_logdiag: torch.Tensor,
    LRinv_tril: torch.Tensor,
    LRinv_logdiag: torch.Tensor,
    LV0inv_tril: torch.Tensor,
    LV0inv_logdiag: torch.Tensor,
    m0: torch.Tensor,
    y: torch.Tensor,
    u_state: Optional[torch.Tensor] = None,
    u_obs: Optional[torch.Tensor] = None,
):
    """
    Computes the sample-wise (e.g. (particle, batch)) EM loss.
    If particle and batch dims are used,
    the ordering is TPBF (time, particle, batch, feature).
    """
    with torch.no_grad():  # Inference (E-Step) is optimal --> analytically zero gradients.
        m, V, Cov = smooth_forward_backward(
            dims=dims,
            A=A,
            B=B,
            C=C,
            D=D,
            LQinv_tril=LQinv_tril,
            LQinv_logdiag=LQinv_logdiag,
            LRinv_tril=LRinv_tril,
            LRinv_logdiag=LRinv_logdiag,
            LV0inv_tril=LV0inv_tril,
            LV0inv_logdiag=LV0inv_logdiag,
            m0=m0,
            y=y,
            u_state=u_state,
            u_obs=u_obs,
        )
    Rinv = inv_from_invcholesky_param(LRinv_tril, LRinv_logdiag)
    Qinv = inv_from_invcholesky_param(LQinv_tril, LQinv_logdiag)
    if (
        A.ndim == 3
    ):  # No Batch and Particle dimension --> Must add at least Batch dimension.
        Rinv, Qinv = Rinv[:, None, :, :], Qinv[:, None, :, :]
        A, B = A[:, None, :, :], B[:, None, :, :]
        C, D = C[:, None, :, :], D[:, None, :, :]
    # initial prior loss
    V0inv = inv_from_invcholesky_param(LV0inv_tril, LV0inv_logdiag)
    dinit = m[0] - m0
    quad_init = matmul(dinit[..., None], dinit[..., None, :]) + V[0]

    loss_init = 0.5 * (
        torch.sum(V0inv * quad_init, dim=(-1, -2))  # FF
        - 2.0 * torch.sum(LV0inv_logdiag, dim=(-1,))  # F
        + dims.state * LOG_2PI
    )

    # transition: Note that we here do no sum the quads over all time-steps.
    b = matvec(B, u_state[:-1]) if u_state is not None else 0
    dtrans = (m[1:] - matvec(A, m[:-1]) - b)[..., None]
    quad_trans = (
        matmul(dtrans, dtrans.transpose(-1, -2))
        + V[1:]
        - matmul(A, Cov[:-1])
        - matmul(Cov[:-1].transpose(-1, -2), A.transpose(-1, -2))
        + matmul(matmul(A, V[:-1]), A.transpose(-1, -2))
    )

    loss_trans = 0.5 * (
        torch.sum(Rinv * quad_trans, dim=(0, -1, -2))  # T...FF
        - 2.0 * torch.sum(LRinv_logdiag, dim=(0, -1))  # T...F
        + (dims.timesteps - 1) * dims.state * LOG_2PI
    )

    # likelihood
    d = matvec(D, u_obs) if u_obs is not None else 0
    dobs = (y - matvec(C, m) - d)[..., None]
    quad_obs = matmul(dobs, dobs.transpose(-1, -2)) + matmul(
        C, matmul(V, C.transpose(-1, -2))
    )
    loss_obs = 0.5 * (
            torch.sum(Qinv * quad_obs, dim=(0, -1, -2))  # T...FF
            - 2.0 * torch.sum(LQinv_logdiag, dim=(0, -1))  # T...F
            + dims.timesteps * dims.target * LOG_2PI
    )

    with torch.no_grad():  # posterior optimal --> entropy has analytically zero gradients as well.
        loss_entropy = -compute_entropy(dims=dims, V=V, Cov=Cov)
    assert (
        loss_trans.shape
        == loss_obs.shape
        == loss_init.shape
        == loss_entropy.shape
    )
    loss_all = loss_trans + loss_obs + loss_init + loss_entropy
    return loss_all


def compute_entropy(
    dims: TensorDims,
    V: torch.Tensor,
    Cov: torch.Tensor,
):
    """
    Compute sample-wise entropy of the smoothing posterior.
    We factorise the smoothing posterior such that the entropy sums over all time-steps
    For t==0, we use the entropy of p(z1 | y_{1:T})

    If particle and/or batch is present, the order is TPBF, i.e. time, particle, batch, feature.
    """
    entropy = 0.0
    for t in range(0, dims.timesteps):
        if t == 0:  # marginal entropy (t==0)
            LVt = cholesky(V[t])
            entropy += 0.5 * (
                2.0 * torch.sum(torch.log(batch_diag(LVt)), dim=(-1,))  # F
                + dims.state * (1.0 + LOG_2PI)
            )
        else:  # Joint entropy (t, t-1) - marginal entropy (t-1)
            Vtm1inv = batch_cholesky_inverse(cholesky(V[t - 1]))
            Cov_cond = V[t] - matmul(
                Cov[t - 1].transpose(-1, -2), matmul(Vtm1inv, Cov[t - 1])
            )
            LCov_cond = cholesky(Cov_cond)
            entropy += 0.5 * (
                2.0
                * torch.sum(torch.log(batch_diag(LCov_cond)), dim=(-1,))  # F
                + dims.state * (1.0 + LOG_2PI)
            )
    return entropy
