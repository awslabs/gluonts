import abc
from dataclasses import dataclass, asdict
from typing import Sequence, Tuple

import torch
from torch import nn

from experiments.model_component_zoo.input_transforms import ControlInputs
from inference.smc.normalize import normalize_log_weights
from inference.smc.resampling import make_criterion_fn_with_ess_threshold, \
    systematic_resampling_indices, resample, make_argmax_log_weights

from models_new_will_replace.dynamical_system import DynamicalSystem, \
    Latents, Prediction
from models.gls_parameters import GLSParameters
from torch_extensions.distributions.parametrised_distribution import \
    ParametrisedMultivariateNormal
from torch_extensions.fusion import ProbabilisticSensorFusion
from utils.utils import list_of_dicts_to_dict_of_list

# TODO: make static_cat without t dim. we just dont index it then.
#  however, in the the Lightning Module.


# TODO: make marginalisation not in sub-modules, but in the reucrrent model.
#  maybe later though.

@dataclass
class RandomVariablesRBSMC:
    """
    Stores either (m, V) or samples or both from a MultivariateNormal.
    We use this instead of torch.distributions.MultivariateNormal in order
    to reduce overhead and increase performance. However,
    performance differences should be tested later, maybe can replace this.
    """

    # Setting default value not possible since subclasses of this dataclass
    # would need to set all fields then with default values too.
    m: (torch.Tensor, None)
    V: (torch.Tensor, None)
    x: (torch.Tensor, None)

    def __post_init__(self):
        has_state_dist_params = tuple(
            param is not None for param in (self.m, self.V)
        )
        if not len(set(has_state_dist_params)) == 1:
            raise Exception("Provide either all or no distribution parameters")

        has_state_sample = self.x is not None
        if not (all(has_state_dist_params) or has_state_sample):
            raise Exception("Provide at least either dist params or samples.")


@dataclass
class LatentsRBSMC(Latents):
    """ Template for models based on Rao-Blackwellized SMC. """

    variables: RandomVariablesRBSMC
    log_weights: torch.Tensor

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


class GaussianLinearSystemRBSMC(DynamicalSystem):
    def __init__(
        self,
        n_state: int,
        n_obs: int,
        n_ctrl_state: int,
        n_ctrl_obs: int,
        n_particle: int,
        gls_base_parameters: GLSParameters,
        state_prior_model: ParametrisedMultivariateNormal,
        obs_encoder: nn.Module,
        resampling_criterion_fn=make_criterion_fn_with_ess_threshold(0.5),
        resampling_indices_fn: callable = systematic_resampling_indices,
    ):
        super().__init__(
            n_state=n_state,
            n_obs=n_obs,
            n_ctrl_state=n_ctrl_state,
            n_ctrl_obs=n_ctrl_obs,  # TODO: rename
        )
        self.n_particle = n_particle

        self.gls_base_parameters = gls_base_parameters
        self.fuse_densities = ProbabilisticSensorFusion()
        self.state_prior_model = state_prior_model
        self.obs_encoder = obs_encoder

        self.resampling_criterion_fn = resampling_criterion_fn
        self.resampling_indices_fn = resampling_indices_fn

    def filter_latent(
        self,
        past_targets: Sequence[torch.Tensor],
        past_controls: (Sequence[ControlInputs], None) = None,
    ) -> Sequence[LatentsRBSMC]:

        n_timesteps = len(past_targets)
        filtered = [None] * n_timesteps
        # TODO: This will fail without inputs. As we access .switch etc.
        #  We need this specific for each class (template) and default Nones.
        #  Alternatively, handle here everywhere if ctrl_t is None.
        controls = (
            [None] * n_timesteps if past_controls is None else past_controls
        )

        for t in range(n_timesteps):
            filtered[t] = self.filter_step(
                lats_tm1=filtered[t - 1] if t > 0 else None,
                tar_t=past_targets[t],
                ctrl_t=controls[t],
            )
        return filtered

    def filter(
        self,
        past_targets: Sequence[torch.Tensor],
        past_controls: (Sequence[ControlInputs], None) = None,
    ) -> Sequence[Prediction]:
        latents_filtered = self.filter_latent(
            past_targets=past_targets, past_controls=past_controls,
        )
        emissions_filtered = [
            self.emit(lats_t=lats, ctrl_t=ctrls)
            for lats, ctrls in zip(latents_filtered, past_controls)
        ]
        predictions = [
            Prediction(latents=l, emissions=e)
            for l, e in zip(latents_filtered, emissions_filtered)
        ]
        return predictions

    def forecast(
        self,
        n_steps_forecast: int,
        initial_latent: LatentsRBSMC,
        future_controls: (Sequence[torch.Tensor], None) = None,
        deterministic=False,
    ) -> Sequence[Prediction]:

        if future_controls is not None:
            assert n_steps_forecast == len(future_controls)

        # Step 1: Re-sample latents before forecast.
        # TODO: This may be made optional and with criterion later.
        resampled_log_norm_weights, resampled_tensors = resample(
            n_particle=self.n_particle,
            log_norm_weights=normalize_log_weights(
                log_weights=initial_latent.log_weights
                if not deterministic
                else make_argmax_log_weights(initial_latent.log_weights),
            ),
            tensors_to_resample=asdict(initial_latent.variables),
            resampling_indices_fn=self.resampling_indices_fn,
            criterion_fn=make_criterion_fn_with_ess_threshold(
                min_ess_ratio=1.0,  # re-sample always / all.
            ),
        )
        # pack re-sampled back into object of our API type.
        resampled_initial_latent = initial_latent.__class__(
            log_weights=resampled_log_norm_weights,  # normalized but thats OK.
            variables=initial_latent.variables.__class__(**resampled_tensors,),
        )

        controls = (
            [None] * n_steps_forecast
            if future_controls is None
            else future_controls
        )
        forecasted = [None] * n_steps_forecast

        for t in range(n_steps_forecast):
            # TODO: currently always uses sample-based forecast.
            #  some models may use mixture / conditional marginal.
            forecasted[t] = self.forecast_sample_step(
                lats_tm1=forecasted[t - 1].latents
                if t > 0
                else resampled_initial_latent,
                ctrl_t=controls[t],
                deterministic=deterministic,
            )
        return forecasted

    def predict(
        self,
        # prediction_length would be misleading as prediction includes past.
        n_steps_forecast: int,
        past_targets: Sequence[torch.Tensor],
        past_controls: (Sequence[ControlInputs], None) = None,
        future_controls: (Sequence[ControlInputs], None) = None,
        deterministic: bool = False,
    ) -> Tuple[Sequence[Prediction], Sequence[Prediction]]:  # past & future
        """
        Predict latent variables and emissions (predictive distribution)
        for both past and future. For past this is filtering (and maybe
        smoothing for a child class) with posterior predictive emissions;
        for future these are probablistic forecasts form unrolling the SSM.
        """

        predictions_filtered = self.filter(
            past_targets=past_targets, past_controls=past_controls,
        )
        predictions_forecast = self.forecast(
            n_steps_forecast=n_steps_forecast,
            initial_latent=predictions_filtered[-1].latents,
            future_controls=future_controls,
            deterministic=deterministic,
        )
        return predictions_filtered, predictions_forecast

    def loss(
        self,
        past_targets: Sequence[torch.Tensor],
        past_controls: (Sequence[ControlInputs], None) = None,
    ) -> torch.Tensor:
        return self.loss_filter(
            past_targets=past_targets,
            past_controls=past_controls,
        )

    def loss_filter(
        self,
        past_targets: Sequence[torch.Tensor],
        past_controls: (Sequence[ControlInputs], None) = None,
    ) -> torch.Tensor:
        """
        Computes an estimate of the negative log marginal likelihood.

        Note: the importance weights exp(log_weights) must be un-normalized
        and correspond to the conditional distributions
        (i.e. incremental importance weights / importance weight updates) s.t.
        their product yields an (unbiased) estimate of the marginal likelihood.
        """
        latents_filtered = self.filter_latent(
            past_targets=past_targets, past_controls=past_controls,
        )
        log_weights = [lats.log_weights for lats in latents_filtered]
        log_conditionals = [torch.logsumexp(lws, dim=0) for lws in log_weights]
        log_marginal = sum(log_conditionals)  # FIVO-type ELBO
        return -log_marginal

    @abc.abstractmethod
    def emit(self, lats_t: LatentsRBSMC, ctrl_t: ControlInputs):
        raise NotImplementedError("must be implemented by child class")

    @abc.abstractmethod
    def filter_step(
        self,
        lats_tm1: (LatentsRBSMC, None),
        tar_t: torch.Tensor,
        ctrl_t: ControlInputs,
    ):
        raise NotImplementedError("must be implemented by child class")

    @abc.abstractmethod
    def forecast_sample_step(
        self,
        lats_tm1: LatentsRBSMC,
        ctrl_t: torch.Tensor,
        deterministic: bool = False,
    ) -> Prediction:
        raise NotImplementedError("must be implemented by child class")

    # @abc.abstractmethod  # Optional.
    def forecast_mixture_step(
        self,
        lats_tm1: LatentsRBSMC,
        ctrl_t: torch.Tensor,
        deterministic: bool = False,
    ) -> Prediction:
        raise NotImplementedError("must be implemented by child class")

    # @abc.abstractmethod
    # def _initial_state(
    #     self,
    #     ctrl_initial: (ControlInputs, None),
    #     n_particle: int,
    #     n_batch: int,
    # ) -> LatentsRBSMC:
    #     raise NotImplementedError("must be implemented by child class")

    # def _split_controls(
    #     self,
    #     controls: (ControlInputs, None),
    #     n_steps_past: int,
    #     n_steps_future: int,
    # ):
    #     n_steps_total = n_steps_past + n_steps_future
    #
    #     if controls is not None:
    #         controls_past = controls[:n_steps_past]
    #         controls_future = controls[n_steps_past:n_steps_total]
    #         assert len(controls_past) == n_steps_past
    #         assert len(controls_future) == n_steps_future
    #     else:
    #         controls_past = None
    #         controls_future = None
    #     return controls_past, controls_future