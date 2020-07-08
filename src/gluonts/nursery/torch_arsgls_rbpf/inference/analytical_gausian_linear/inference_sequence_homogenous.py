import torch
from torch import matmul
from torch_extensions.ops import (
    batch_cholesky_inverse,
    batch_diag,
    kron,
    symmetrize,
    matvec,
    cov_from_invcholesky_param,
    inv_from_invcholesky_param,
    cholesky,
)
from inference.analytical_gausian_linear.inference_step import (
    filter_forward_prediction_step,
    filter_forward_measurement_step,
    smooth_backward_step,
)
from utils.utils import add_sample_dims_to_initial_state, LOG_2PI


def filter_forward(
    dims,
    A,
    B,
    C,
    D,
    LQinv_tril,
    LQinv_logdiag,
    LRinv_tril,
    LRinv_logdiag,
    LV0inv_tril,
    LV0inv_logdiag,
    m0,
    y,
    u_state=None,
    u_obs=None,
):
    device, dtype = A.device, A.dtype

    R = cov_from_invcholesky_param(LRinv_tril, LRinv_logdiag)
    Q = cov_from_invcholesky_param(LQinv_tril, LQinv_logdiag)
    V0 = cov_from_invcholesky_param(LV0inv_tril, LV0inv_logdiag)
    m0, V0 = add_sample_dims_to_initial_state(m0=m0, V0=V0, dims=dims)

    # pre-compute biases
    b = matvec(B, u_state) if u_state is not None else 0
    d = matvec(D, u_obs) if u_obs is not None else 0

    m_fw = torch.zeros(
        (dims.timesteps, dims.batch, dims.state), device=device, dtype=dtype
    )
    V_fw = torch.zeros(
        (dims.timesteps, dims.batch, dims.state, dims.state),
        device=device,
        dtype=dtype,
    )
    for t in range(0, dims.timesteps):
        (mp, Vp) = (
            filter_forward_prediction_step(
                m=m_fw[t - 1], V=V_fw[t - 1], R=R, A=A, b=b[t - 1],
            )
            if t > 0
            else (m0, V0)
        )
        m_fw[t], V_fw[t] = filter_forward_measurement_step(
            y=y[t], m=mp, V=Vp, Q=Q, C=C, d=d[t]
        )
    return m_fw, V_fw


def smooth_forward_backward(
    dims,
    A,
    B,
    C,
    D,
    LQinv_tril,
    LQinv_logdiag,
    LRinv_tril,
    LRinv_logdiag,
    LV0inv_tril,
    LV0inv_logdiag,
    m0,
    y,
    u_state=None,
    u_obs=None,
):
    device, dtype = A.device, A.dtype
    R = cov_from_invcholesky_param(LRinv_tril, LRinv_logdiag)

    # pre-compute biases
    b = matvec(B, u_state) if u_state is not None else 0
    m_sm = torch.zeros(
        (dims.timesteps, dims.batch, dims.state), device=device, dtype=dtype
    )
    V_sm = torch.zeros(
        (dims.timesteps, dims.batch, dims.state, dims.state),
        device=device,
        dtype=dtype,
    )
    Cov_sm = torch.zeros(
        (dims.timesteps, dims.batch, dims.state, dims.state),
        device=device,
        dtype=dtype,
    )

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
        m_sm[t], V_sm[t], Cov_sm[t] = smooth_backward_step(
            m_sm=m_sm[t + 1],
            V_sm=V_sm[t + 1],
            m_fw=m_fw[t],
            V_fw=V_fw[t],
            A=A,
            R=R,
            b=b[t],
        )
    return m_sm, V_sm, Cov_sm


def smooth_global(
    dims,
    A,
    B,
    C,
    D,
    LQinv_tril,
    LQinv_logdiag,
    LRinv_tril,
    LRinv_logdiag,
    LV0inv_tril,
    LV0inv_logdiag,
    m0,
    y,
    u_state=None,
    u_obs=None,
):
    """ compute posterior by direct inversion of unrolled model """
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

    # pre-compute biases
    b = matvec(B, u_state) if u_state is not None else 0
    d = matvec(D, u_obs) if u_obs is not None else 0

    Rinv = symmetrize(torch.cholesky_inverse(cholesky(R)))
    Qinv = symmetrize(torch.cholesky_inverse(cholesky(Q)))
    V0inv = symmetrize(torch.cholesky_inverse(cholesky(V0)))

    CtQinvymd = matvec(matmul(C.transpose(-1, -2), Qinv), y - d)
    h_obs = CtQinvymd.transpose(1, 0).reshape(
        (dims.batch, dims.timesteps * dims.state,)
    )
    Q_obs = kron(
        torch.eye(dims.timesteps, dtype=dtype, device=device),
        matmul(C.transpose(-1, -2), matmul(Qinv, C)),
    )

    AtRinvA = matmul(A.transpose(-1, -2), matmul(Rinv, A))
    RinvA = matmul(Rinv, A)

    h_field[:, : dims.state] = matmul(V0inv, m0).repeat(
        (dims.batch,) + (1,) * (h_field.ndim - 1)
    )
    Q_field[:, : dims.state, : dims.state] += V0inv.repeat(
        (dims.batch,) + (1,) * (Q_field.ndim - 1)
    )
    for t in range(dims.timesteps - 1):
        idx = t * dims.state
        h_field[:, idx : idx + dims.state] += -matvec(
            RinvA.transpose(-1, -2), b[t]
        )
        h_field[:, idx + dims.state : idx + 2 * dims.state] += matvec(
            Rinv, b[t]
        )
        Q_field[:, idx : idx + dims.state, idx : idx + dims.state] += AtRinvA
        Q_field[
            :, idx : idx + dims.state, idx + dims.state : idx + 2 * dims.state
        ] += -RinvA.transpose(-1, -2)
        Q_field[
            :, idx + dims.state : idx + 2 * dims.state, idx : idx + dims.state
        ] += -RinvA
        Q_field[
            :,
            idx + dims.state : idx + 2 * dims.state,
            idx + dims.state : idx + 2 * dims.state,
        ] += Rinv

    L_all_inv = torch.inverse(cholesky(Q_field + Q_obs))
    V_all = matmul(L_all_inv.transpose(-1, -2), L_all_inv)
    m_all = matvec(V_all, h_obs + h_field)

    # Pytorch has no Fortran style reading of indices.
    m = m_all.reshape((dims.batch, dims.timesteps, dims.state)).transpose(0, 1)
    V, Cov = [], []
    for t in range(0, dims.timesteps):
        idx = t * dims.state
        V.append(V_all[:, idx : idx + dims.state, idx : idx + dims.state])
        if t < (dims.timesteps - 1):
            Cov.append(
                V_all[
                    :,
                    idx : idx + dims.state,
                    idx + dims.state : idx + 2 * dims.state,
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
    dims,
    A,
    B,
    C,
    D,
    LQinv_tril,
    LQinv_logdiag,
    LRinv_tril,
    LRinv_logdiag,
    LV0inv_tril,
    LV0inv_logdiag,
    m0,
    u_state=None,
    u_obs=None,
):
    device, dtype = A.device, A.dtype

    # generate noise
    wz = torch.randn(dims.timesteps, dims.batch, dims.state)
    wy = torch.randn(dims.timesteps, dims.batch, dims.target)

    # pre-compute cholesky matrices
    LR = torch.inverse(
        torch.tril(LRinv_tril, -1) + torch.diag(torch.exp(LRinv_logdiag))
    )
    LQ = torch.inverse(
        torch.tril(LQinv_tril, -1) + torch.diag(torch.exp(LQinv_logdiag))
    )
    LV0 = torch.inverse(
        torch.tril(LV0inv_tril, -1) + torch.diag(torch.exp(LV0inv_logdiag))
    )

    # pre-compute biases
    b = matvec(B, u_state) if u_state is not None else 0
    d = matvec(D, u_obs) if u_obs is not None else 0

    # Initial step.
    # Note: We cannot use in-place operations here because we must backprop through y.
    x = [m0 + matvec(LV0, wz[0])] + [None] * (dims.timesteps - 1)
    y = [matvec(C, x[0]) + d[0] + matvec(LQ, wy[0])] + [None] * (
        dims.timesteps - 1
    )
    for t in range(1, dims.timesteps):
        x[t] = matvec(A, x[t - 1]) + b[t - 1] + matvec(LR, wz[t])
        y[t] = matvec(C, x[t]) + d[t] + matvec(LQ, wy[t])
    x = torch.stack(x, dim=0)
    y = torch.stack(y, dim=0)
    return x, y


def loss_forward(
    dims,
    A,
    B,
    C,
    D,
    LQinv_tril,
    LQinv_logdiag,
    LRinv_tril,
    LRinv_logdiag,
    LV0inv_tril,
    LV0inv_logdiag,
    m0,
    y,
    u_state=None,
    u_obs=None,
):
    device, dtype = A.device, A.dtype

    R = cov_from_invcholesky_param(LRinv_tril, LRinv_logdiag)
    Q = cov_from_invcholesky_param(LQinv_tril, LQinv_logdiag)
    V0 = cov_from_invcholesky_param(LV0inv_tril, LV0inv_logdiag)
    m0, V0 = add_sample_dims_to_initial_state(m0=m0, V0=V0, dims=dims)

    # pre-compute biases
    b = matvec(B, u_state) if u_state is not None else 0
    d = matvec(D, u_obs) if u_obs is not None else 0

    # Note: We can not use (more readable) in-place operations due to backprop problems.
    m_fw = [None] * dims.timesteps
    V_fw = [None] * dims.timesteps
    loss = torch.zeros((dims.batch,), device=device, dtype=dtype)
    for t in range(0, dims.timesteps):
        (mp, Vp) = (
            filter_forward_prediction_step(
                m=m_fw[t - 1], V=V_fw[t - 1], R=R, A=A, b=b[t - 1],
            )
            if t > 0
            else (m0, V0)
        )
        m_fw[t], V_fw[t], dobs_norm, LVpyinv = filter_forward_measurement_step(
            y=y[t], m=mp, V=Vp, Q=Q, C=C, d=d[t], return_loss_components=True
        )
        loss += (
                0.5 * torch.sum(dobs_norm ** 2, dim=-1)
                - 0.5 * 2 * torch.sum(torch.log(batch_diag(LVpyinv)), dim=(-1,))
                + 0.5 * dims.target * LOG_2PI
        )

    return loss


def loss_em(
    dims,
    A,
    B,
    C,
    D,
    LQinv_tril,
    LQinv_logdiag,
    LRinv_tril,
    LRinv_logdiag,
    LV0inv_tril,
    LV0inv_logdiag,
    m0,
    y,
    u_state=None,
    u_obs=None,
):
    Rinv = inv_from_invcholesky_param(LRinv_tril, LRinv_logdiag)
    Qinv = inv_from_invcholesky_param(LQinv_tril, LQinv_logdiag)

    with torch.no_grad():  # E-Step is optimal --> analytically zero gradients.
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
        loss_entropy = -compute_entropy(dims=dims, V=V, Cov=Cov)

    Cov_sum = torch.sum(Cov[:-1], dim=0)  # idx -1 is Cov_{T, T+1}.
    V_sum = torch.sum(V, dim=0)
    V_sum_head = V_sum - V[-1]
    V_sum_tail = V_sum - V[0]

    # initial prior loss
    V0inv = inv_from_invcholesky_param(LV0inv_tril, LV0inv_logdiag)
    delta_init = m[0] - m0
    quad_init = matmul(delta_init[..., None], delta_init[..., None, :]) + V[0]
    loss_init = 0.5 * (
        torch.sum(V0inv * quad_init, dim=(-1, -2))
        - 2.0 * torch.sum(LV0inv_logdiag)
        + dims.state * LOG_2PI
    )

    # transition losses - summed over all time-steps
    b = matvec(B, u_state[:-1]) if u_state is not None else 0
    delta_trans = m[1:] - matvec(A, m[:-1]) - b
    quad_trans = (
        matmul(
            delta_trans.transpose(0, 1).transpose(-1, -2),
            delta_trans.transpose(0, 1),
        )
        + V_sum_tail
        - matmul(A, Cov_sum)
        - matmul(Cov_sum.transpose(-1, -2), A.transpose(-1, -2))
        + matmul(matmul(A, V_sum_head), A.transpose(-1, -2))
    )
    loss_trans = 0.5 * (
        torch.sum(Rinv * quad_trans, dim=(-1, -2))
        - 2.0
        * (dims.timesteps - 1)
        * torch.sum(LRinv_logdiag, dim=-1)
        + (dims.timesteps - 1) * dims.state * LOG_2PI
    )

    # observation losses - summed over all time-steps
    d = matvec(D, u_obs) if u_obs is not None else 0
    delta_obs = y - matvec(C, m) - d
    quad_obs = matmul(
        delta_obs.transpose(0, 1).transpose(-1, -2), delta_obs.transpose(0, 1)
    ) + matmul(C, matmul(V_sum, C.transpose(-1, -2)))
    loss_obs = 0.5 * (
            torch.sum(Qinv * quad_obs, dim=(-1, -2))
            - 2.0 * dims.timesteps * torch.sum(LQinv_logdiag, dim=-1)
            + dims.timesteps * dims.target * LOG_2PI
    )

    loss = loss_trans + loss_obs + loss_init + loss_entropy
    return loss


def compute_entropy(dims, V, Cov):
    """ Compute entropy of Gaussian posterior from E-step (in Markovian SSM) """
    entropy = 0.0
    for t in range(0, dims.timesteps):
        if t == 0:  # marginal entropy (t==0)
            LVt = cholesky(V[t])
            entropy += 0.5 * 2.0 * torch.sum(
                torch.log(batch_diag(LVt)), dim=(-1,)
            ) + 0.5 * dims.state * (1 + LOG_2PI)
        else:  # joint entropy (t, t-1) - marginal entropy (t-1)
            Vtm1inv = batch_cholesky_inverse(cholesky(V[t - 1]))
            Cov_cond = V[t] - matmul(
                Cov[t - 1].transpose(-1, -2), matmul(Vtm1inv, Cov[t - 1])
            )
            LCov_cond = cholesky(Cov_cond)

            entropy += 0.5 * 2.0 * torch.sum(
                torch.log(batch_diag(LCov_cond)), dim=(-1,)
            ) + 0.5 * dims.state * (1.0 + LOG_2PI)
    return entropy
