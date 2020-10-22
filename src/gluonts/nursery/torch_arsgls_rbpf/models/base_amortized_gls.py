import abc
from dataclasses import dataclass, asdict
from typing import Sequence, Tuple, Optional, Union

import torch
from torch import nn
from torch.distributions import MultivariateNormal

from models.base_gls import (
    BaseGaussianLinearSystem,
    Latents,
    Prediction,
    ControlInputs,
)
from models.gls_parameters.gls_parameters import GLSParameters
from torch_extensions.distributions.parametrised_distribution import (
    ParametrisedMultivariateNormal,
)
from torch_extensions.fusion import ProbabilisticSensorFusion
from torch_extensions.ops import cholesky
from utils.utils import list_of_dicts_to_dict_of_list


@dataclass
class LatentsRBSMC(Latents):
    """ Template for models based on Rao-Blackwellized SMC. """

    log_weights: torch.Tensor

    # TODO: Still need this?

    @classmethod
    def sequence_to_tensor(cls, sequence: Sequence, dim=0):
        """
        Class method to turn a Sequence of LatentsRBSMC objects
        into a single LatentsRBSMC object,
        where each log_weights and each tensor in variables are stacked
        on the time-dimension.

        If subclasses only add more tensors to the variables field,
        this class can be used in child classes,
        otherwise it should be overwritten.
        This method should raise an error in that case though.
        """
        raise Exception(
            "Do not use this yet. Not properly tested. "
            "Focusing on other things first. "
        )

        assert len(sequence) > 0, "must provide at least one item."
        lat_cls = sequence[0].__class__
        var_cls = sequence[0].variables.__class__

        # make sure we
        seq_fields = set(sequence[0].__dict__.keys())
        if not seq_fields == {"variables", "log_weights"}:
            raise Exception(
                f"sequence items have unexpected fields: {seq_fields})."
                "subclasses that have a different API should "
                "overwrite method sequence_to_tensor."
            )

        assert all(
            [
                lat_cls == sequence[idx].__class__
                for idx in range(len(sequence))
            ]
        )

        variables_list_of_dicts = [
            asdict(sequence[idx].variables) for idx in range(len(sequence))
        ]
        variables_dict_of_list = list_of_dicts_to_dict_of_list(
            variables_list_of_dicts
        )
        variables_dict_of_stacked = {
            key: torch.stack(val_list)
            if not any(v is None for v in val_list)
            else None
            for key, val_list in variables_dict_of_list.items()
        }
        return lat_cls(
            log_weights=torch.stack(
                [sequence[idx].log_weights for idx in range(len(sequence))],
                dim=dim,
            ),
            variables=var_cls(**variables_dict_of_stacked),
        )


class BaseAmortizedGaussianLinearSystem(BaseGaussianLinearSystem):
    def __init__(
        self,
        n_state: int,
        n_target: int,
        n_ctrl_state: int,
        n_ctrl_target: int,
        n_particle: int,
        gls_base_parameters: GLSParameters,
        state_prior_model: ParametrisedMultivariateNormal,
        encoder: nn.Module,
    ):
        super().__init__(
            n_state=n_state,
            n_target=n_target,
            n_ctrl_state=n_ctrl_state,
            n_ctrl_target=n_ctrl_target,
        )
        self.n_particle = n_particle
        self.gls_base_parameters = gls_base_parameters
        self.state_prior_model = state_prior_model
        self.encoder = encoder
        self.fuse_densities = ProbabilisticSensorFusion()

    def predict(
        self,
        # prediction_length would be misleading as prediction includes past.
        n_steps_forecast: int,
        past_targets: [Sequence[torch.Tensor], torch.Tensor],
        past_controls: Optional[
            Union[Sequence[ControlInputs], ControlInputs]
        ] = None,
        future_controls: Optional[Sequence[ControlInputs]] = None,
        deterministic: bool = False,
        smooth_past: bool = False,
    ) -> Tuple[Sequence[Prediction], Sequence[Prediction]]:  # past & future
        """
        Predict latent variables and emissions (predictive distribution)
        for both past and future. For past this is filtering or smoothing
        with posterior predictive emissions;
        for future these are probabilistic forecasts from unrolling the SSM.
        """
        if smooth_past:
            latents_inferred = self.smooth(
                past_targets=past_targets, past_controls=past_controls,
            )
        else:
            latents_inferred = self.filter(
                past_targets=past_targets, past_controls=past_controls,
            )

        emission_dist_inferred = [
            self.emit(lats_t=latents_inferred[t], ctrl_t=past_controls[t])
            for t in range(len(latents_inferred))
        ]
        emissions_inferred = [
            e.mean if deterministic else e.sample()
            for e in emission_dist_inferred
        ]
        predictions_inferred = [
            Prediction(latents=l, emissions=e)
            for l, e in zip(latents_inferred, emissions_inferred)
        ]

        predictions_forecast = self.forecast(
            n_steps_forecast=n_steps_forecast,
            initial_latent=latents_inferred[-1],
            future_controls=future_controls,
            deterministic=deterministic,
        )
        return predictions_inferred, predictions_forecast

    def forecast(
        self,
        n_steps_forecast: int,
        initial_latent: Latents,
        future_controls: Optional[
            Union[Sequence[ControlInputs], ControlInputs]
        ] = None,
        deterministic: bool = False,
    ) -> Sequence[Prediction]:

        # TODO: we only support sample forecasts atm
        #  The metrics such as CRPS in GluonTS are evaluated with samples only.
        #  Some models could retain states closed-form though.
        if initial_latent.variables.x is None:
            initial_latent.variables.x = MultivariateNormal(
                loc=initial_latent.variables.m,
                scale_tril=cholesky(initial_latent.variables.V),
            ).rsample()
            initial_latent.variables.m = None
            initial_latent.variables.V = None
            initial_latent.variables.Cov = None

        initial_latent, future_controls = self._prepare_forecast(
            initial_latent=initial_latent,
            controls=future_controls,
            deterministic=deterministic,
        )

        return self._sample_trajectory_from_initial(
            n_steps_forecast=n_steps_forecast,
            initial_latent=initial_latent,
            future_controls=future_controls,
            deterministic=deterministic,
        )

    def sample_generative(
        self,
        n_steps_forecast: int,
        n_batch: int,
        n_particle: int,
        future_controls: Optional[Sequence[ControlInputs]] = None,
        deterministic=False,
        **kwargs,
    ) -> Sequence[Prediction]:

        initial_latent = self._sample_initial_latents(
            n_particle=n_particle, n_batch=n_batch,
        )

        return self._sample_trajectory_from_initial(
            n_steps_forecast=n_steps_forecast,
            initial_latent=initial_latent,
            future_controls=future_controls,
            deterministic=deterministic,
        )

    def _sample_trajectory_from_initial(
        self,
        n_steps_forecast: int,
        initial_latent: Latents,
        future_controls: Optional[
            Union[Sequence[ControlInputs], ControlInputs]
        ] = None,
        deterministic: bool = False,
    ) -> Sequence[Prediction]:
        # initial_latent is considered t == -1

        if future_controls is not None:
            assert n_steps_forecast == len(future_controls)

        controls = (
            [None] * n_steps_forecast
            if future_controls is None
            else future_controls
        )
        samples = [None] * n_steps_forecast

        for t in range(n_steps_forecast):
            samples[t] = self.sample_step(
                lats_tm1=samples[t - 1].latents if t > 0 else initial_latent,
                ctrl_t=controls[t],
                deterministic=deterministic,
            )
        return samples

    def filter(
        self,
        past_targets: [Sequence[torch.Tensor], torch.Tensor],
        past_controls: Optional[
            Union[Sequence[ControlInputs], ControlInputs]
        ] = None,
        past_targets_is_observed: Optional[
            Union[Sequence[torch.Tensor], torch.Tensor]
        ] = None,
    ) -> Sequence[Latents]:

        n_timesteps = len(past_targets)
        controls = (
            [None] * n_timesteps if past_controls is None else past_controls
        )
        filtered = [None] * n_timesteps

        for t in range(n_timesteps):
            filtered[t] = self.filter_step(
                lats_tm1=filtered[t - 1] if t > 0 else None,
                tar_t=past_targets[t],
                ctrl_t=controls[t],
                tar_is_obs_t=past_targets_is_observed[t]
                if past_targets_is_observed is not None
                else None,
            )
        return filtered

    def smooth(
        self,
        past_targets: [Sequence[torch.Tensor], torch.Tensor],
        past_controls: Optional[
            Union[Sequence[ControlInputs], ControlInputs]
        ] = None,
        past_targets_is_observed: Optional[
            Union[Sequence[torch.Tensor], torch.Tensor]
        ] = None,
    ) -> Sequence[Latents]:
        """ Forward-Backward Smoothing (Rauch-Tung-Striebel) """

        n_timesteps = len(past_targets)
        smoothed = [None] * n_timesteps

        filtered = self.filter(
            past_targets=past_targets,
            past_controls=past_controls,
            past_targets_is_observed=past_targets_is_observed,
        )

        smoothed[-1] = filtered[
            -1
        ]  # start backward recursion from last filter
        smoothed[-1].variables.Cov = torch.zeros_like(smoothed[-1].variables.V)
        for t in reversed(range(n_timesteps - 1)):
            smoothed[t] = self.smooth_step(
                lats_smooth_tp1=smoothed[t + 1], lats_filter_t=filtered[t],
            )
        return smoothed

    @abc.abstractmethod
    def emit(
        self, lats_t: Latents, ctrl_t: ControlInputs,
    ) -> torch.distributions.Distribution:
        raise NotImplementedError("must be implemented by child class")

    @abc.abstractmethod
    def filter_step(
        self,
        lats_tm1: (Latents, None),
        tar_t: torch.Tensor,
        ctrl_t: ControlInputs,
        tar_is_obs_t: Optional[torch.Tensor] = None,
    ) -> Latents:
        raise NotImplementedError("must be implemented by child class")

    @abc.abstractmethod
    def smooth_step(
        self, lats_smooth_tp1: (Latents, None), lats_filter_t: (Latents, None),
    ) -> Latents:
        raise NotImplementedError("must be implemented by child class")

    @abc.abstractmethod
    def sample_step(
        self,
        lats_tm1: Latents,
        ctrl_t: torch.Tensor,
        deterministic: bool = False,
    ) -> Prediction:
        raise NotImplementedError("must be implemented by child class")

    @abc.abstractmethod
    def _sample_initial_latents(self, n_particle, n_batch) -> Latents:
        raise NotImplementedError("must be implemented by child class")

    @abc.abstractmethod
    def _prepare_forecast(
        self,
        initial_latent: Latents,
        controls: Optional[
            Union[Sequence[ControlInputs], ControlInputs]
        ] = None,
        deterministic: bool = False,
    ):
        raise NotImplementedError("Should be implemented by child class")
