from experiments import model_component_zoo

SyntheticGLSParameters = model_component_zoo.gls_parameters.GlsParametersSeasonalityISSM
SyntheticStateToSwitchEncoder = lambda config: None
SyntheticObsToSwitchEncoder = lambda config: None
SyntheticStatePriorModel = model_component_zoo.state_priors.StatePriorModelNoInputs
SyntheticSwitchTransitionModel = model_component_zoo.switch_transitions \
    .SwitchTransitionModelGaussian
SyntheticSwitchPriorModel = model_component_zoo.switch_priors \
    .SwitchPriorModelGaussian
SyntheticInputTransform = model_component_zoo.input_transforms.InputTransformMLP
