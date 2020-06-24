from box import Box
import torch

from models.homogenous_gaussian_lineary_system import (
    GaussianLinearSystemHomogenous,
)
from utils.utils import make_dummy_ssm_params, make_dummy_input_data
from utils.local_seed import local_seed


def _make_model_and_data(device, dtype, n_timesteps=10, n_data=20, seed=42):
    with local_seed(seed=seed):
        true_params = make_dummy_ssm_params()
        input_data = make_dummy_input_data(
            ssm_params=true_params, n_timesteps=n_timesteps, n_data=n_data
        )
        true_model = GaussianLinearSystemHomogenous(
            n_obs=1, n_state=2, n_ctrl_state=1, n_ctrl_obs=3,
        )
        x, y = true_model.sample(**input_data)
        data = Box(**input_data, y=y)

    model = (
        GaussianLinearSystemHomogenous(
            n_obs=1, n_state=2, n_ctrl_state=1, n_ctrl_obs=3,
        )
        .to(device)
        .to(dtype)
    )
    data = Box({name: val.to(dtype).to(device) for name, val in data.items()})
    return model, data


def _test_inference_identical(
    tolerance=2e-5, dtype=torch.float32, device="cuda"
):
    model, data = _make_model_and_data(device=device, dtype=dtype)

    # Use unrolled SSM ("global") and standard inference as reference.
    m_gl, V_gl, Cov_gl = model.smooth_global(**data)
    m_fb, V_fb, Cov_fb = model.smooth_forward_backward(**data)

    # Compute max absolute error of fw-bw vs. global smoothing.
    err = torch.max(torch.abs(m_fb - m_gl))
    assert err <= tolerance, f"large error: {err}"
    err = torch.max(torch.abs(V_fb - V_gl))
    assert err <= tolerance, f"large error: {err}"
    err = torch.max(torch.abs(Cov_fb - Cov_gl))
    assert err <= tolerance, f"large error: {err}"

    return True


def _test_loss_identical(tolerance=2e-5, dtype=torch.float32, device="cuda:0"):
    model, data = _make_model_and_data(device=device, dtype=dtype)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-3)

    optimizer.zero_grad()
    loss_fw = model.loss_forward(**data)
    loss_fw.backward(retain_graph=True)
    grads_fw = (p.grad for p in model.parameters())

    optimizer.zero_grad()
    loss_em = model.loss_em(**data)
    loss_em.backward()
    grads_em = (p.grad for p in model.parameters())

    err = (loss_em - loss_fw).abs().max()
    assert err <= tolerance, f"large error: {err}"

    err = max(
        (g_em - g_fw).abs().max() for g_em, g_fw in zip(grads_em, grads_fw)
    )
    assert err <= tolerance, f"large error: {err}"
    return True


def test_inference_identical_cpu_float64(tolerance=1e-10):
    _test_inference_identical(
        tolerance=tolerance, device="cpu", dtype=torch.float64
    )


def test_inference_identical_gpu_float32(tolerance=1e-3):
    _test_inference_identical(
        tolerance=tolerance, device="cuda", dtype=torch.float64
    )


def test_loss_identical_cpu_float64(tolerance=1e-10):
    _test_loss_identical(
        tolerance=tolerance, device="cpu", dtype=torch.float64
    )


def test_loss_identical_gpu_float32(tolerance=1e-10):
    _test_loss_identical(
        tolerance=tolerance, device="cuda", dtype=torch.float64
    )
