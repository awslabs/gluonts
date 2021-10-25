import torch.nn as nn

from .subnetworks import (
    InitialContinuousDistribution,
    ContinuousStateTransition,
    InitialDiscreteDistribution,
    DiscreteStateTransition,
    GaussianDistributionOutput,
    RNNInferenceNetwork,
    SelectIndex,
    RawControlToFeat,
    ControlToNSTF,
    TransformerEmbedder,
)
from .model import SNLDS, REDSDS


def build_model(config):
    (
        emission_network,
        posterior_rnn,
        posterior_mlp,
        x0_networks,
        x_transition_networks,
        z0_network,
        z_transition_network,
        embedding_network,
        ctrl_feat_network,
        duration_network
    ) = get_network_instances(config)

    x_init = InitialContinuousDistribution(
        networks=x0_networks,
        dist_dim=config['x_dim'],
        num_categories=config['num_categories'],
        use_tied_cov=config['initial_state']['tied_cov'],
        use_trainable_cov=config['initial_state']['trainable_cov'],
        sigma=config['initial_state']['fixed_sigma'],
        takes_ctrl=config['control']['x'],
        max_scale=config['initial_state'].get(
            'max_scale', 1.0
        ),
        scale_nonlinearity=config['initial_state'].get(
            'scale_nonlinearity', 'softplus'
        )
    )

    x_transition = ContinuousStateTransition(
        transition_networks=x_transition_networks,
        dist_dim=config['x_dim'],
        num_categories=config['num_categories'],
        use_tied_cov=config['continuous_transition']['tied_cov'],
        use_trainable_cov=config['continuous_transition']['trainable_cov'],
        sigma=config['continuous_transition']['fixed_sigma'],
        takes_ctrl=config['control']['x'],
        max_scale=config['continuous_transition'].get(
            'max_scale', 1.0
        ),
        scale_nonlinearity=config['continuous_transition'].get(
            'scale_nonlinearity', 'softplus'
        )
    )

    z_init = InitialDiscreteDistribution(
        network=z0_network,
        num_categories=config['num_categories'],
        takes_ctrl=config['control']['z'])

    z_transition = DiscreteStateTransition(
        transition_network=z_transition_network,
        num_categories=config['num_categories'],
        takes_ctrl=config['control']['z'],
        takes_x=config['discrete_transition']['takes_x'],
        takes_y=config['discrete_transition']['takes_y'])

    emission_network = GaussianDistributionOutput(
        network=emission_network,
        dist_dim=config['obs_dim'],
        use_tied_cov=config['emission']['tied_cov'],
        use_trainable_cov=config['emission']['trainable_cov'],
        sigma=config['emission']['fixed_sigma'],
        max_scale=config['emission'].get(
            'max_scale', 1.0
        ),
        scale_nonlinearity=config['emission'].get(
            'scale_nonlinearity', 'softplus'
        )
    )

    posterior_dist = GaussianDistributionOutput(
        network=posterior_mlp,
        dist_dim=config['x_dim'],
        use_tied_cov=config['inference']['tied_cov'],
        use_trainable_cov=config['inference']['trainable_cov'],
        sigma=config['inference']['fixed_sigma'],
        max_scale=config['inference'].get(
            'max_scale', 1.0
        ),
        scale_nonlinearity=config['inference'].get(
            'scale_nonlinearity', 'softplus'
        )
    )

    posterior_network = RNNInferenceNetwork(
        posterior_rnn=posterior_rnn,
        posterior_dist=posterior_dist,
        x_dim=config['x_dim'],
        embedding_network=embedding_network,
        takes_ctrl=config['control']['inference'])

    rawctrl2feat_network = None
    if config['control']['has_ctrl']:
        assert ctrl_feat_network is not None
        rawctrl2feat_network = RawControlToFeat(
            ctrl_feat_network,
            n_staticfeat=config['control']['n_staticfeat'],
            n_timefeat=config['control']['n_timefeat'],
            embedding_dim=config['control']['emb_dim'])

    if config['model'] == 'REDSDS':
        assert duration_network is not None
        d_min = config.get('d_min', 1)
        ctrl2nstf_network = ControlToNSTF(
            duration_network,
            config['num_categories'],
            config['d_max'],
            d_min=d_min)

    if config['model'] == 'SNLDS':
        model = SNLDS(
            x_init=x_init,
            continuous_transition_network=x_transition,
            z_init=z_init,
            discrete_transition_network=z_transition,
            emission_network=emission_network,
            inference_network=posterior_network,
            ctrl_transformer=rawctrl2feat_network,
            continuous_state_dim=config['x_dim'],
            num_categories=config['num_categories'],
            context_length=config['context_length'],
            prediction_length=config['prediction_length'],
            discrete_state_prior=None,
            transform_target=config['transform_target'],
            transform_only_scale=config['transform_only_scale'],
            use_jacobian=config['use_jacobian'],)
    elif config['model'] == 'REDSDS':
        model = REDSDS(
            x_init=x_init,
            continuous_transition_network=x_transition,
            z_init=z_init,
            discrete_transition_network=z_transition,
            emission_network=emission_network,
            inference_network=posterior_network,
            ctrl_transformer=rawctrl2feat_network,
            ctrl2nstf_network=ctrl2nstf_network,
            continuous_state_dim=config['x_dim'],
            num_categories=config['num_categories'],
            d_max=config['d_max'],
            context_length=config['context_length'],
            prediction_length=config['prediction_length'],
            discrete_state_prior=None,
            transform_target=config['transform_target'],
            transform_only_scale=config['transform_only_scale'],
            use_jacobian=config['use_jacobian'],)
    return model


def get_network_instances(config):

    if config['experiment'] == 'bouncing_ball'\
            or config['experiment'] == '3modesystem'\
            or config['experiment'] == 'bee'\
            or config['experiment'] == 'gts_univariate':
        # Inference Network
        inf_ctrl_dim = config['control']['feat_dim']\
            if config['control']['inference'] else 0
        inf_out_dim = config['x_dim']
        if not config['inference']['tied_cov']:
            inf_out_dim *= 2
        if config['inference']['embedder'] == 'brnn':
            embedding_network = nn.Sequential(
                nn.GRU(
                    input_size=config['obs_dim'],
                    hidden_size=config['inference']['embedding_rnndim'],
                    num_layers=config['inference']['embedding_rnnlayers'],
                    batch_first=True,
                    bidirectional=True),
                SelectIndex(index=0))
            # 2 * because bidirectional
            emb_out_dim = 2 * config['inference']['embedding_rnndim']
        elif config['inference']['embedder'] == 'transformer':
            embedding_network = TransformerEmbedder(
                obs_dim=config['obs_dim'],
                emb_dim=config['inference']['embedding_trans_embdim'],
                use_pe=config['inference']['embedding_trans_usepe'],
                nhead=config['inference']['embedding_trans_nhead'],
                dim_feedforward=config['inference']['embedding_trans_mlpdim'],
                n_layers=config['inference']['embedding_trans_nlayers']
            )
            # 2 * because of positional encoding
            emb_out_dim = 2 * config['inference']['embedding_trans_embdim']
        else:
            embedding_network = None
            emb_out_dim = config['obs_dim']

        posterior_rnn = None
        if config['inference']['use_causal_rnn']:
            posterior_rnn = nn.RNNCell(
                    emb_out_dim
                    + config['x_dim'] + inf_ctrl_dim,
                    config['inference']['causal_rnndim'])

        posterior_mlp_in_dim = config['inference']['causal_rnndim']\
            if config['inference']['use_causal_rnn'] else (
                emb_out_dim + inf_ctrl_dim)
        posterior_mlp = nn.Sequential(
            nn.Linear(
                posterior_mlp_in_dim,
                config['inference']['mlp_hiddendim']),
            nn.ReLU(True),
            nn.Linear(
                config['inference']['mlp_hiddendim'],
                inf_out_dim))

        # Emission Network
        if config['emission']['model_type'] == 'linear':
            emission_network = nn.Sequential(
                nn.Linear(config['x_dim'], config['obs_dim'], False))
        else:
            if config['dataset'] == 'bee':
                emission_network = nn.Sequential(
                    nn.Linear(config['x_dim'], 32),
                    nn.ReLU(True),
                    nn.Linear(32, 64),
                    nn.ReLU(True),
                    nn.Linear(64, config['obs_dim']))
            else:
                emission_network = nn.Sequential(
                    nn.Linear(config['x_dim'], 8),
                    nn.ReLU(True),
                    nn.Linear(8, 32),
                    nn.ReLU(True),
                    nn.Linear(32, config['obs_dim']))

        # Initial Continuous State
        if config['control']['has_ctrl'] and config['control']['x']:
            x0_networks = [nn.Sequential(
                nn.Linear(
                    config['control']['feat_dim'],
                    config['initial_state']['mlp_hiddendim']),
                nn.ReLU(True),
                nn.Linear(
                    config['initial_state']['mlp_hiddendim'],
                    config['x_dim']))
                for _ in range(config['num_categories'])]
        else:
            x0_networks = [nn.Sequential(
                nn.Linear(1, config['x_dim'], bias=False))
                for _ in range(config['num_categories'])]

        # Continuous Transition
        x_ctrl_dim = config['control']['feat_dim']\
            if config['control']['x'] else 0
        x_trans_out_dim = config['x_dim']
        if not config['continuous_transition']['tied_cov']:
            x_trans_out_dim *= 2
        if config['continuous_transition']['model_type'] == 'linear':
            x_transition_networks = [nn.Sequential(
                nn.Linear(config['x_dim'] + x_ctrl_dim,
                          x_trans_out_dim,
                          bias=False)
                )
                for _ in range(config['num_categories'])]
        else:
            x_transition_networks = [nn.Sequential(
                nn.Linear(
                    config['x_dim'] + x_ctrl_dim,
                    config['continuous_transition']['mlp_hiddendim']),
                nn.ReLU(True),
                nn.Linear(
                    config['continuous_transition']['mlp_hiddendim'],
                    x_trans_out_dim))
                for _ in range(config['num_categories'])]

        # Initial Discrete State
        if config['control']['has_ctrl'] and config['control']['z']:
            z0_network = nn.Sequential(
                nn.Linear(
                    config['control']['feat_dim'],
                    config['initial_switch']['mlp_hiddendim']),
                nn.ReLU(True),
                nn.Linear(
                    config['initial_switch']['mlp_hiddendim'],
                    config['num_categories']))
        else:
            z0_network = nn.Linear(
                1, config['num_categories'], bias=False)

        # Discrete Transition
        discrete_in_dim = config['x_dim']\
            if config['discrete_transition']['takes_x'] else 0
        discrete_in_dim += config['obs_dim']\
            if config['discrete_transition']['takes_y'] else 0
        z_ctrl_dim = config['control']['feat_dim']\
            if config['control']['z'] else 0
        if discrete_in_dim + z_ctrl_dim > 0:
            z_transition_network = nn.Sequential(
                nn.Linear(
                    discrete_in_dim + z_ctrl_dim,
                    4 * (config['num_categories'] ** 2)),
                nn.ReLU(True),
                nn.Linear(4 * (config['num_categories'] ** 2),
                          config['num_categories'] ** 2))
        else:
            z_transition_network = nn.Linear(
                1, config['num_categories'] ** 2, bias=False)
            print('[*] No recurrence!')

        # Control Transformer
        ctrl_feat_network = None
        if config['control']['has_ctrl']:
            n_input_feats = config['control']['emb_dim'] +\
                config['control']['n_timefeat']
            ctrl_feat_network = nn.Sequential(
                nn.Linear(
                    n_input_feats,
                    config['control']['mlp_hiddendim']),
                nn.ReLU(True),
                nn.Linear(
                    config['control']['mlp_hiddendim'],
                    config['control']['feat_dim'])
            )

        # Control to Duration (NSTF)
        duration_network = None
        if 'd_max' in config:
            if config['control']['has_ctrl']:
                duration_network = nn.Sequential(
                    nn.Linear(config['control']['feat_dim'], 64),
                    nn.ReLU(True),
                    nn.Linear(
                        64, config['num_categories'] * config['d_max'])
                )
            else:
                duration_network = nn.Linear(
                    1, config['num_categories'] * config['d_max'], bias=False)
    else:
        raise ValueError(f"Unknown experiment: {config['experiment']}!")
    return (
        emission_network,
        posterior_rnn,
        posterior_mlp,
        x0_networks,
        x_transition_networks,
        z0_network,
        z_transition_network,
        embedding_network,
        ctrl_feat_network,
        duration_network
    )
