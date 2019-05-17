# Standard library imports
from itertools import chain, combinations

# Third-party imports
import mxnet as mx
import pytest

# First-party imports
from gluonts.block.feature import FeatureAssembler, FeatureEmbedder

# fix the seed for the tests in this file
mx.random.seed(78_431_519)


@pytest.mark.parametrize("hybridize", [False, True])
@pytest.mark.parametrize(
    "config",
    (
        lambda N, T: [
            # single static feature
            dict(
                shape=(N, 1),
                kwargs=dict(cardinalities=[50], embedding_dims=[10]),
            ),
            # single dynamic feature
            dict(
                shape=(N, T, 1),
                kwargs=dict(cardinalities=[2], embedding_dims=[10]),
            ),
            # multiple static features
            dict(
                shape=(N, 4),
                kwargs=dict(
                    cardinalities=[50, 50, 50, 50],
                    embedding_dims=[10, 20, 30, 40],
                ),
            ),
            # multiple dynamic features
            dict(
                shape=(N, T, 3),
                kwargs=dict(
                    cardinalities=[30, 30, 30], embedding_dims=[10, 20, 30]
                ),
            ),
        ]
    )(10, 20),
)
def test_feature_embedder(config, hybridize):
    out_shape = config['shape'][:-1] + (
        sum(config['kwargs']['embedding_dims']),
    )

    embed_feature = FeatureEmbedder(
        prefix='embed_feature_', **config['kwargs']
    )
    embed_feature.collect_params().initialize(mx.initializer.One())

    if hybridize:
        embed_feature.hybridize()

    def test_parameters_length():
        exp_params_len = len(embed_feature.collect_params().keys())
        act_params_len = len(config['kwargs']['embedding_dims'])
        assert exp_params_len == act_params_len

    def test_parameter_names():
        for param in embed_feature.collect_params():
            assert param.startswith('embed_feature_')

    def test_forward_pass():
        act_output = embed_feature(mx.nd.ones(shape=config['shape']))
        exp_output = mx.nd.ones(shape=out_shape)

        assert act_output.shape == exp_output.shape
        assert mx.nd.sum(act_output - exp_output) < 1e-20

    test_parameters_length()
    test_parameter_names()
    test_forward_pass()


@pytest.mark.parametrize("hybridize", [False, True])
@pytest.mark.parametrize(
    "config",
    (
        lambda N, T: [
            dict(
                N=N,
                T=T,
                static_cat=dict(C=2),
                static_real=dict(C=5),
                dynamic_cat=dict(C=3),
                dynamic_real=dict(C=4),
                embed_static=dict(
                    cardinalities=[2, 4],
                    embedding_dims=[3, 6],
                    prefix='static_cat_',
                ),
                embed_dynamic=dict(
                    cardinalities=[30, 30, 30],
                    embedding_dims=[10, 20, 30],
                    prefix='dynamic_cat_',
                ),
            )
        ]
    )(10, 25),
)
def test_feature_assembler(config, hybridize):
    # iterate over the power-set of all possible feature types, excluding the empty set
    feature_types = {
        'static_cat',
        'static_real',
        'dynamic_cat',
        'dynamic_real',
    }
    feature_combs = chain.from_iterable(
        combinations(feature_types, r)
        for r in range(1, len(feature_types) + 1)
    )

    # iterate over the power-set of all possible feature types, including the empty set
    embedder_types = {'embed_static', 'embed_dynamic'}
    embedder_combs = chain.from_iterable(
        combinations(embedder_types, r)
        for r in range(0, len(embedder_types) + 1)
    )

    for enabled_embedders in embedder_combs:
        embed_static = (
            FeatureEmbedder(**config['embed_static'])
            if 'embed_static' in enabled_embedders
            else None
        )
        embed_dynamic = (
            FeatureEmbedder(**config['embed_dynamic'])
            if 'embed_dynamic' in enabled_embedders
            else None
        )

        for enabled_features in feature_combs:
            assemble_feature = FeatureAssembler(
                T=config['T'],
                # use_static_cat='static_cat' in enabled_features,
                # use_static_real='static_real' in enabled_features,
                # use_dynamic_cat='dynamic_cat' in enabled_features,
                # use_dynamic_real='dynamic_real' in enabled_features,
                embed_static=embed_static,
                embed_dynamic=embed_dynamic,
            )

            assemble_feature.collect_params().initialize(mx.initializer.One())

            if hybridize:
                assemble_feature.hybridize()

            def test_parameters_length():
                exp_params_len = sum(
                    [
                        len(config[k]['embedding_dims'])
                        for k in ['embed_static', 'embed_dynamic']
                        if k in enabled_embedders
                    ]
                )
                act_params_len = len(assemble_feature.collect_params().keys())
                assert exp_params_len == act_params_len

            def test_parameter_names():
                if embed_static:
                    for param in embed_static.collect_params():
                        assert param.startswith('static_cat_')
                if embed_dynamic:
                    for param in embed_dynamic.collect_params():
                        assert param.startswith('dynamic_cat_')

            def test_forward_pass():
                N, T = config['N'], config['T']

                inp_features = []
                out_features = []

                if 'static_cat' not in enabled_features:
                    inp_features.append(mx.nd.zeros(shape=(N, 1)))
                    out_features.append(mx.nd.zeros(shape=(N, T, 1)))
                elif embed_static:  # and 'static_cat' in enabled_features
                    C = config['static_cat']['C']
                    inp_features.append(
                        mx.nd.concat(
                            *[
                                mx.nd.random.uniform(
                                    0,
                                    config['embed_static']['cardinalities'][c],
                                    shape=(N, 1),
                                ).floor()
                                for c in range(C)
                            ],
                            dim=1,
                        )
                    )
                    out_features.append(
                        mx.nd.ones(
                            shape=(
                                N,
                                T,
                                sum(config['embed_static']['embedding_dims']),
                            )
                        )
                    )
                else:  # not embed_static and 'static_cat' in enabled_features
                    C = config['static_cat']['C']
                    inp_features.append(
                        mx.nd.concat(
                            *[
                                mx.nd.random.uniform(
                                    0,
                                    config['embed_static']['cardinalities'][c],
                                    shape=(N, 1),
                                ).floor()
                                for c in range(C)
                            ],
                            dim=1,
                        )
                    )
                    out_features.append(
                        mx.nd.tile(
                            mx.nd.expand_dims(inp_features[-1], axis=1),
                            reps=(1, T, 1),
                        )
                    )

                if 'static_real' not in enabled_features:
                    inp_features.append(mx.nd.zeros(shape=(N, 1)))
                    out_features.append(mx.nd.zeros(shape=(N, T, 1)))
                else:
                    C = config['static_real']['C']
                    static_real = mx.nd.random.uniform(0, 100, shape=(N, C))
                    inp_features.append(static_real)
                    out_features.append(
                        mx.nd.tile(
                            static_real.expand_dims(axis=-2), reps=(1, T, 1)
                        )
                    )

                if 'dynamic_cat' not in enabled_features:
                    inp_features.append(mx.nd.zeros(shape=(N, T, 1)))
                    out_features.append(mx.nd.zeros(shape=(N, T, 1)))
                elif embed_dynamic:  # and 'static_cat' in enabled_features
                    C = config['dynamic_cat']['C']
                    inp_features.append(
                        mx.nd.concat(
                            *[
                                mx.nd.random.uniform(
                                    0,
                                    config['embed_dynamic']['cardinalities'][
                                        c
                                    ],
                                    shape=(N, T, 1),
                                ).floor()
                                for c in range(C)
                            ],
                            dim=2,
                        )
                    )
                    out_features.append(
                        mx.nd.ones(
                            shape=(
                                N,
                                T,
                                sum(config['embed_dynamic']['embedding_dims']),
                            )
                        )
                    )
                else:  # not embed_dynamic and 'dynamic_cat' in enabled_features
                    C = config['dynamic_cat']['C']
                    inp_features.append(
                        mx.nd.concat(
                            *[
                                mx.nd.random.uniform(
                                    0,
                                    config['embed_dynamic']['cardinalities'][
                                        c
                                    ],
                                    shape=(N, T, 1),
                                ).floor()
                                for c in range(C)
                            ],
                            dim=2,
                        )
                    )
                    out_features.append(inp_features[-1])

                if 'dynamic_real' not in enabled_features:
                    inp_features.append(mx.nd.zeros(shape=(N, T, 1)))
                    out_features.append(mx.nd.zeros(shape=(N, T, 1)))
                else:
                    C = config['dynamic_real']['C']
                    dynamic_real = mx.nd.random.uniform(
                        0, 100, shape=(N, T, C)
                    )
                    inp_features.append(dynamic_real)
                    out_features.append(dynamic_real)

                exp_output = mx.nd.concat(*out_features, dim=2)
                act_output = assemble_feature(*inp_features)

                assert exp_output.shape == act_output.shape
                assert mx.nd.sum(exp_output - act_output) < 1e-20

            test_parameters_length()
            test_parameter_names()
            test_forward_pass()
