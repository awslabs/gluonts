import torch
from inference.analytical_gausian_linear.inference_sequence_inhomogenous import \
    filter_forward, smooth_forward_backward, smooth_global, sample, loss_em, \
    loss_forward
from models.dynamical_system import DynamicalSystem
from models.homogenous_gaussian_lineary_system import \
    ornstein_uhlenbeck_initialization


class GaussianLinearSystemInhomogenous(DynamicalSystem):
    """
    GLS with inhomogenous (i.e. time-dependent) dynamics.

    Contains optionally a batch and particle dimension for the SSM parameters
    (default None).
    The purupose of this class is mainly for testing and sanity check,
    no practical purposes.
    """

    def __init__(self, n_state, n_obs, n_ctrl_state, n_ctrl_obs, n_timesteps,
                 n_batch=None, n_particle=None,
                 initialization_fn=ornstein_uhlenbeck_initialization):
        super().__init__(n_state=n_state, n_obs=n_obs,
                         n_ctrl_state=n_ctrl_state, n_ctrl_obs=n_ctrl_obs)
        self.n_timesteps = n_timesteps
        self.n_batch = n_batch
        self.n_particle = n_particle

        params = initialization_fn(
            n_obs=self.n_obs,
            n_state=self.n_state,
            n_ctrl_state=self.n_ctrl_state,
            n_ctrl_obs=self.n_ctrl_obs,
        )

        # 1x initial prior parameters
        self.m0 = torch.nn.Parameter(params.m0)
        self.LV0inv_tril = torch.nn.Parameter(params.LV0inv_tril)
        self.LV0inv_logdiag = torch.nn.Parameter(params.LV0inv_logdiag)

        # T-1 transition parameters
        dims_batch = tuple(
            [n_batch]) if n_batch is not None and n_batch > 0 else tuple()
        dims_particle = tuple([n_particle]) \
            if n_particle is not None and n_particle > 0 else tuple()
        dynamics_dims = (self.n_timesteps - 1,) + dims_particle + dims_batch
        measurement_dims = (self.n_timesteps,) + dims_particle + dims_batch

        self.A = torch.nn.Parameter(params.A.repeat(
            dynamics_dims + (1,) * params.A.ndim))
        self.B = torch.nn.Parameter(params.B.repeat(
            dynamics_dims + (1,) * params.B.ndim))
        self.LRinv_tril = torch.nn.Parameter(params.LRinv_tril.repeat(
            dynamics_dims + (1,) * params.LRinv_tril.ndim))
        self.LRinv_logdiag = torch.nn.Parameter(params.LRinv_logdiag.repeat(
            dynamics_dims + (1,) * params.LRinv_logdiag.ndim))

        # T measurement parameters
        self.C = torch.nn.Parameter(params.C.repeat(
            measurement_dims + (1,) * params.C.ndim))
        self.D = torch.nn.Parameter(params.D.repeat(
            measurement_dims + (1,) * params.D.ndim))
        self.LQinv_tril = torch.nn.Parameter(params.LQinv_tril.repeat(
            measurement_dims + (1,) * params.LQinv_tril.ndim))
        self.LQinv_logdiag = torch.nn.Parameter(params.LQinv_logdiag.repeat(
            measurement_dims + (1,) * params.LQinv_logdiag.ndim))

    def filter_forward(self, y, u_state=None, u_obs=None):
        dims = self.get_dims(y)
        return filter_forward(
            dims=dims, A=self.A, B=self.B, C=self.C, D=self.D,
            LQinv_tril=self.LQinv_tril, LQinv_logdiag=self.LQinv_logdiag,
            LRinv_tril=self.LRinv_tril, LRinv_logdiag=self.LRinv_logdiag,
            LV0inv_tril=self.LV0inv_tril, LV0inv_logdiag=self.LV0inv_logdiag,
            m0=self.m0,
            y=y, u_state=u_state, u_obs=u_obs,
        )

    def smooth_forward_backward(self, y, u_state=None, u_obs=None):
        dims = self.get_dims(y)
        return smooth_forward_backward(
            dims=dims, A=self.A, B=self.B, C=self.C, D=self.D,
            LQinv_tril=self.LQinv_tril, LQinv_logdiag=self.LQinv_logdiag,
            LRinv_tril=self.LRinv_tril, LRinv_logdiag=self.LRinv_logdiag,
            LV0inv_tril=self.LV0inv_tril, LV0inv_logdiag=self.LV0inv_logdiag,
            m0=self.m0,
            y=y, u_state=u_state, u_obs=u_obs,
        )

    def smooth_global(self, y, u_state=None, u_obs=None):
        dims = self.get_dims(y)
        return smooth_global(
            dims=dims, A=self.A, B=self.B, C=self.C, D=self.D,
            LQinv_tril=self.LQinv_tril, LQinv_logdiag=self.LQinv_logdiag,
            LRinv_tril=self.LRinv_tril, LRinv_logdiag=self.LRinv_logdiag,
            LV0inv_tril=self.LV0inv_tril, LV0inv_logdiag=self.LV0inv_logdiag,
            m0=self.m0,
            y=y, u_state=u_state, u_obs=u_obs,
        )

    def sample(self, u_state=None, u_obs=None, n_timesteps=None, n_batch=None):
        dims = self.get_dims(u_state=u_state, u_obs=u_obs,
                             n_timesteps=n_timesteps, n_batch=n_batch)
        return sample(
            dims=dims, A=self.A, B=self.B, C=self.C, D=self.D,
            LQinv_tril=self.LQinv_tril, LQinv_logdiag=self.LQinv_logdiag,
            LRinv_tril=self.LRinv_tril, LRinv_logdiag=self.LRinv_logdiag,
            LV0inv_tril=self.LV0inv_tril, LV0inv_logdiag=self.LV0inv_logdiag,
            m0=self.m0,
            u_state=u_state, u_obs=u_obs,
        )

    def loss_forward(self, y, u_state=None, u_obs=None):
        dims = self.get_dims(y)
        loss = loss_forward(
            dims=dims, A=self.A, B=self.B, C=self.C, D=self.D,
            LQinv_tril=self.LQinv_tril, LQinv_logdiag=self.LQinv_logdiag,
            LRinv_tril=self.LRinv_tril, LRinv_logdiag=self.LRinv_logdiag,
            LV0inv_tril=self.LV0inv_tril, LV0inv_logdiag=self.LV0inv_logdiag,
            m0=self.m0,
            y=y, u_state=u_state, u_obs=u_obs,
        )
        if dims.particle is not None and dims.particle > 0:
            loss = loss.mean(dim=0)
        if dims.batch is not None and dims.batch > 0:
            loss = loss.sum(dim=0)
        return loss

    def loss_em(self, y, u_state=None, u_obs=None):
        dims = self.get_dims(y)
        loss = loss_em(
            dims=dims, A=self.A, B=self.B, C=self.C, D=self.D,
            LQinv_tril=self.LQinv_tril, LQinv_logdiag=self.LQinv_logdiag,
            LRinv_tril=self.LRinv_tril, LRinv_logdiag=self.LRinv_logdiag,
            LV0inv_tril=self.LV0inv_tril, LV0inv_logdiag=self.LV0inv_logdiag,
            m0=self.m0,
            y=y, u_state=u_state, u_obs=u_obs,
        )
        if dims.particle is not None and dims.particle > 0:
            loss = loss.mean(dim=0)
        if dims.batch is not None and dims.batch > 0:
            loss = loss.sum(dim=0)
        return loss
