from box import Box
import torch

from models_new_will_replace.gls_inhomogenous import (
    GaussianLinearSystemInhomogenous,
)
from models_new_will_replace.gls_homogenous import (
    GaussianLinearSystemHomogenous,
)
from utils.utils import make_dummy_ssm_params, make_dummy_input_data
from utils.local_seed import local_seed


def _make_models_and_data(device, dtype, n_data=20, n_timesteps=10, seed=42):
    with local_seed(seed=seed):
        true_params = make_dummy_ssm_params()
        controls = make_dummy_input_data(
            ssm_params=true_params, n_timesteps=n_timesteps, n_data=n_data
        )
        true_model = GaussianLinearSystemHomogenous(
            n_target=1, n_state=2, n_ctrl_state=1, n_ctrl_target=3,
        )
        samples = true_model.sample(
            n_steps_forecast=n_timesteps,
            n_batch=n_data,
            future_controls=controls,
            deterministic=False,
        )
        emissions = torch.stack([sample.emissions for sample in samples], dim=0)
        data = Box(
            past_controls=controls,
            past_targets=emissions,
        )

    model_homogenous = GaussianLinearSystemHomogenous(
        n_target=1, n_state=2, n_ctrl_state=1, n_ctrl_target=3,
    )
    model_inhomogenous = GaussianLinearSystemInhomogenous(
        n_target=1,
        n_state=2,
        n_ctrl_state=1,
        n_ctrl_target=3,
        n_timesteps=n_timesteps,
    )
    model_inhomogenous_batch = GaussianLinearSystemInhomogenous(
        n_target=1,
        n_state=2,
        n_ctrl_state=1,
        n_ctrl_target=3,
        n_timesteps=n_timesteps,
        n_batch=n_data,
    )

    # hack
    model_inhomogenous.A = torch.nn.Parameter(model_inhomogenous_batch.A[:, 0])
    model_inhomogenous.B = torch.nn.Parameter(model_inhomogenous_batch.B[:, 0])
    model_inhomogenous.C = torch.nn.Parameter(model_inhomogenous_batch.C[:, 0])
    model_inhomogenous.D = torch.nn.Parameter(model_inhomogenous_batch.D[:, 0])
    model_inhomogenous.LRinv_logdiag = torch.nn.Parameter(
        model_inhomogenous_batch.LRinv_logdiag[:, 0]
    )
    model_inhomogenous.LRinv_tril = torch.nn.Parameter(
        model_inhomogenous_batch.LRinv_tril[:, 0]
    )
    model_inhomogenous.LQinv_logdiag = torch.nn.Parameter(
        model_inhomogenous_batch.LQinv_logdiag[:, 0]
    )
    model_inhomogenous.LQinv_tril = torch.nn.Parameter(
        model_inhomogenous_batch.LQinv_tril[:, 0]
    )
    model_inhomogenous.LV0inv_logdiag = torch.nn.Parameter(
        model_inhomogenous_batch.LV0inv_logdiag
    )
    model_inhomogenous.LV0inv_tril = torch.nn.Parameter(
        model_inhomogenous_batch.LV0inv_tril
    )
    model_inhomogenous.m0 = torch.nn.Parameter(model_inhomogenous_batch.m0)

    model_homogenous.A = torch.nn.Parameter(model_inhomogenous.A[0])
    model_homogenous.B = torch.nn.Parameter(model_inhomogenous.B[0])
    model_homogenous.C = torch.nn.Parameter(model_inhomogenous.C[0])
    model_homogenous.D = torch.nn.Parameter(model_inhomogenous.D[0])
    model_homogenous.LRinv_logdiag = torch.nn.Parameter(
        model_inhomogenous.LRinv_logdiag[0]
    )
    model_homogenous.LRinv_tril = torch.nn.Parameter(
        model_inhomogenous.LRinv_tril[0]
    )
    model_homogenous.LQinv_logdiag = torch.nn.Parameter(
        model_inhomogenous.LQinv_logdiag[0]
    )
    model_homogenous.LQinv_tril = torch.nn.Parameter(
        model_inhomogenous.LQinv_tril[0]
    )
    model_homogenous.LV0inv_logdiag = torch.nn.Parameter(
        model_inhomogenous.LV0inv_logdiag
    )
    model_homogenous.LV0inv_tril = torch.nn.Parameter(
        model_inhomogenous.LV0inv_tril
    )
    model_homogenous.m0 = torch.nn.Parameter(model_inhomogenous.m0)

    model_homogenous = model_homogenous.to(device).to(dtype)
    model_inhomogenous = model_inhomogenous.to(device).to(dtype)
    model_inhomogenous_batch = model_inhomogenous_batch.to(device).to(dtype)
    data = Box({name: val.to(dtype).to(device) for name, val in data.items()})
    return model_inhomogenous_batch, model_inhomogenous, model_homogenous, data


def _test_inference_identical_to_homogenous(
    tolerance=2e-5,
    dtype=torch.float32,
    device="cuda",
    n_data=20,
    n_timesteps=10,
):
    torch.manual_seed(42)
    (
        model_inhomogenous_batch,
        model_inhomogenous,
        model_homogenous,
        data,
    ) = _make_models_and_data(
        device=device, dtype=dtype, n_data=n_data, n_timesteps=n_timesteps
    )

    (
        m_fb_ib,
        V_fb_ib,
        Cov_fb_ib,
    ) = model_inhomogenous_batch._smooth_forward_backward(**data)
    m_gl_ib, V_gl_ib, Cov_gl_ib = model_inhomogenous_batch._smooth_global(
        **data
    )

    m_fb_i, V_fb_i, Cov_fb_i = model_inhomogenous._smooth_forward_backward(
        **data
    )

    m_fb_h, V_fb_h, Cov_fb_h = model_homogenous._smooth_forward_backward(**data)
    m_gl_h, V_gl_h, Cov_gl_h = model_homogenous._smooth_global(**data)

    # fw-bw: inhomogenous and batch-individual vs. just inhomogenous
    err = torch.max(torch.abs(m_fb_ib - m_fb_i))
    assert err <= tolerance, f"large error: {err}"
    err = torch.max(torch.abs(V_fb_ib - V_fb_i))
    assert err <= tolerance, f"large error: {err}"
    err = torch.max(torch.abs(Cov_fb_ib - Cov_fb_i))
    assert err <= tolerance, f"large error: {err}"

    # fw-bw: in-homogenous vs homogenous
    err = torch.max(torch.abs(m_fb_i - m_fb_h))
    assert err <= tolerance, f"large error: {err}"
    err = torch.max(torch.abs(V_fb_i - V_fb_h))
    assert err <= tolerance, f"large error: {err}"
    err = torch.max(torch.abs(Cov_fb_i - Cov_fb_h))
    assert err <= tolerance, f"large error: {err}"

    # global: inhomogenous and batch-individual vs. homogenous
    err = torch.max(torch.abs(m_gl_ib - m_gl_h))
    assert err <= tolerance, f"large error: {err}"
    err = torch.max(torch.abs(V_gl_ib - V_gl_h))
    assert err <= tolerance, f"large error: {err}"
    err = torch.max(torch.abs(Cov_gl_ib - Cov_gl_h))
    assert err <= tolerance, f"large error: {err}"

    return True


def _test_loss_identical_to_homogenous(
    tolerance=2e-5,
    dtype=torch.float32,
    device="cuda:0",
    n_data=20,
    n_timesteps=10,
):
    (
        model_inhomogenous_batch,
        model_inhomogenous,
        model_homogenous,
        data,
    ) = _make_models_and_data(
        device=device, dtype=dtype, n_data=n_data, n_timesteps=n_timesteps
    )

    loss_fw_inhomogenous_batch = model_inhomogenous_batch._loss_forward(**data)
    loss_fw_inhomogenous = model_inhomogenous._loss_forward(**data)
    loss_fw_homogenous = model_homogenous._loss_forward(**data)
    err = (loss_fw_inhomogenous - loss_fw_homogenous).abs().max()
    assert err <= tolerance, f"large error: {err}"
    err = (loss_fw_inhomogenous - loss_fw_inhomogenous_batch).abs().max()
    assert err <= tolerance, f"large error: {err}"

    loss_em_inhomogenous_batch = model_inhomogenous_batch._loss_em(**data)
    loss_em_inhomogenous = model_inhomogenous._loss_em(**data)
    loss_em_homogenous = model_homogenous._loss_em(**data)
    err = (loss_em_inhomogenous - loss_em_homogenous).abs().max()
    assert err <= tolerance, f"large error: {err}"
    err = (loss_em_inhomogenous - loss_em_inhomogenous_batch).abs().max()
    assert err <= tolerance, f"large error: {err}"

    return True


def test_inference_identical_to_homogenous_cpu_float64(tolerance=1e-20):
    _test_inference_identical_to_homogenous(
        tolerance=tolerance, device="cpu", dtype=torch.float64,
    )


def test_inference_identical_to_homogenous_gpu_float32(tolerance=1e-4):
    _test_inference_identical_to_homogenous(
        tolerance=tolerance, device="cuda", dtype=torch.float32,
    )


def test_loss_identical_to_homogenous_cpu_float64(tolerance=1e-12):
    _test_loss_identical_to_homogenous(
        tolerance=tolerance, device="cpu", dtype=torch.float64,
    )


def test_loss_identical_to_homogenous_gpu_float32(tolerance=1e-4):
    _test_loss_identical_to_homogenous(
        tolerance=tolerance, device="cuda", dtype=torch.float64,
    )
