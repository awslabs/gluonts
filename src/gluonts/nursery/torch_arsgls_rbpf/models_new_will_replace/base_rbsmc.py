import abc
from dataclasses import dataclass, asdict
from typing import Sequence, Tuple, Optional, Union

import torch
from torch import nn

from inference.smc.normalize import normalize_log_weights
from inference.smc.resampling import make_criterion_fn_with_ess_threshold, \
    systematic_resampling_indices, resample, make_argmax_log_weights

from models_new_will_replace.dynamical_system import DynamicalSystem, \
    Latents, Prediction, ControlInputs
from models_new_will_replace.gls_parameters.gls_parameters import GLSParameters
from torch_extensions.distributions.parametrised_distribution import \
    ParametrisedMultivariateNormal
from torch_extensions.fusion import ProbabilisticSensorFusion
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


class BaseRBSMC(DynamicalSystem):
    def __init__(
        self,
        n_state: int,
        n_target: int,
        n_ctrl_state: int,
        n_ctrl_target: int,
        n_particle: int,
        gls_base_parameters: GLSParameters,
        state_prior_model: ParametrisedMultivariateNormal,
        obs_encoder: nn.Module,
        resampling_criterion_fn=make_criterion_fn_with_ess_threshold(0.5),
        resampling_indices_fn: callable = systematic_resampling_indices,
    ):
        super().__init__(
            n_state=n_state,
            n_target=n_target,
            n_ctrl_state=n_ctrl_state,
            n_ctrl_target=n_ctrl_target,
        )
        self.n_particle = n_particle

        self.gls_base_parameters = gls_base_parameters
        self.fuse_densities = ProbabilisticSensorFusion()
        self.state_prior_model = state_prior_model
        self.obs_encoder = obs_encoder

        self.resampling_criterion_fn = resampling_criterion_fn
        self.resampling_indices_fn = resampling_indices_fn

    # def filter(
    #     self,
    #     past_targets: [Sequence[torch.Tensor], torch.Tensor],
    #     past_controls: Optional[Union[Sequence[ControlInputs], ControlInputs]] = None,
    # ) -> Sequence[Prediction]:
    #     latents_filtered = self.filter_latent(
    #         past_targets=past_targets, past_controls=past_controls,
    #     )
    #     emissions_filtered = [
    #         self.emit(lats_t=lats, ctrl_t=ctrls)
    #         for lats, ctrls in zip(latents_filtered, past_controls)
    #     ]
    #     predictions = [
    #         Prediction(latents=l, emissions=e)
    #         for l, e in zip(latents_filtered, emissions_filtered)
    #     ]
    #     return predictions

    def forecast(
            self,
            n_steps_forecast: int,
            initial_latent: LatentsRBSMC,
            future_controls: Optional[Union[Sequence[ControlInputs], ControlInputs]] = None,
            deterministic=False,
    ) -> Sequence[Prediction]:

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
            # TODO: these are normalized. its OK as its actually not used
            #  but what is a better design here?
            log_weights=resampled_log_norm_weights,
            variables=initial_latent.variables.__class__(**resampled_tensors,),
        )

        return self._sample_forecast(
            n_steps_forecast=n_steps_forecast,
            initial_latent=resampled_initial_latent,
            future_controls=future_controls,
            deterministic=deterministic,
        )

    def sample(
            self,
            n_steps_forecast: int,
            n_batch: int,
            n_particle: int,
            future_controls: Optional[Sequence[torch.Tensor]] = None,
            deterministic=False,
            **kwargs,
    ) -> Sequence[Prediction]:
        initial_latent = self._sample_initial_latents(
            n_particle=n_particle, n_batch=n_batch,
        )

        return self._sample_forecast(
            n_steps_forecast=n_steps_forecast,
            initial_latent=initial_latent,
            future_controls=future_controls,
            deterministic=deterministic,
        )

    def filter(
        self,
        past_targets: [Sequence[torch.Tensor], torch.Tensor],
        past_controls: Optional[Union[Sequence[ControlInputs], ControlInputs]] = None,
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

    def predict(
        self,
        # prediction_length would be misleading as prediction includes past.
        n_steps_forecast: int,
        past_targets: [Sequence[torch.Tensor], torch.Tensor],
        past_controls: Optional[Union[Sequence[ControlInputs], ControlInputs]] = None,
        future_controls: Optional[Sequence[ControlInputs]] = None,
        deterministic: bool = False,
    ) -> Tuple[Sequence[Prediction], Sequence[Prediction]]:  # past & future
        """
        Predict latent variables and emissions (predictive distribution)
        for both past and future. For past this is filtering (and maybe
        smoothing for a child class) with posterior predictive emissions;
        for future these are probablistic forecasts form unrolling the SSM.
        """

        latents_filtered = self.filter(
            past_targets=past_targets, past_controls=past_controls,
        )
        emissions_filtered = [
            self.emit(lats_t=lats, ctrl_t=ctrls)
            for lats, ctrls in zip(latents_filtered, past_controls)
        ]
        predictions_filtered = [
            Prediction(latents=l, emissions=e)
            for l, e in zip(latents_filtered, emissions_filtered)
        ]

        predictions_forecast = self.forecast(
            n_steps_forecast=n_steps_forecast,
            initial_latent=predictions_filtered[-1].latents,
            future_controls=future_controls,
            deterministic=deterministic,
        )
        return predictions_filtered, predictions_forecast

    def loss(
        self,
        past_targets: [Sequence[torch.Tensor], torch.Tensor],
        past_controls: Optional[Union[Sequence[ControlInputs], ControlInputs]] = None,
    ) -> torch.Tensor:
        return self.loss_filter(
            past_targets=past_targets,
            past_controls=past_controls,
        )

    def loss_filter(
        self,
        past_targets: [Sequence[torch.Tensor], torch.Tensor],
        past_controls: Optional[Union[Sequence[ControlInputs], ControlInputs]] = None,
    ) -> torch.Tensor:
        """
        Computes an estimate of the negative log marginal likelihood.

        Note: the importance weights exp(log_weights) must be un-normalized
        and correspond to the conditional distributions
        (i.e. incremental importance weights / importance weight updates) s.t.
        their product yields an (unbiased) estimate of the marginal likelihood.
        """
        latents_filtered = self.filter(
            past_targets=past_targets, past_controls=past_controls,
        )
        log_weights = [lats.log_weights for lats in latents_filtered]
        log_conditionals = [torch.logsumexp(lws, dim=0) for lws in log_weights]
        log_marginal = sum(log_conditionals)  # FIVO-type ELBO
        return -log_marginal

    def _mixture_forecast(
            self,
            n_steps_forecast: int,
            initial_latent: LatentsRBSMC,
            future_controls: Optional[Sequence[ControlInputs]] = None,
            deterministic=False,
    ) -> Sequence[Prediction]:
        raise NotImplementedError("TODO")

    def _sample_forecast(
        self,
        n_steps_forecast: int,
        initial_latent: LatentsRBSMC,
        future_controls: Optional[Sequence[torch.Tensor]] = None,
        deterministic=False,
    ) -> Sequence[Prediction]:

        if future_controls is not None:
            assert n_steps_forecast == len(future_controls)

        controls = (
            [None] * n_steps_forecast
            if future_controls is None
            else future_controls
        )
        forecasted = [None] * n_steps_forecast

        for t in range(n_steps_forecast):
            forecasted[t] = self.forecast_sample_step(
                lats_tm1=forecasted[t - 1].latents if t > 0 else initial_latent,
                ctrl_t=controls[t],
                deterministic=deterministic,
            )
        return forecasted

    @abc.abstractmethod
    def _sample_initial_latents(self, n_particle, n_batch) -> LatentsRBSMC:
        raise NotImplementedError("must be implemented by child class")

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

