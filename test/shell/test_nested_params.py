from gluonts.shell.sagemaker.nested_params import decode_nested_parameters


def test_nested_params():
    data = decode_nested_parameters(
        {
            "$env.num_workers": "4",
            "$evaluation.quantiles": [0.1, 0.5, 0.9],
            "prediction_length": 14,
        }
    )

    hps = data.pop("")
    assert hps["prediction_length"] == 14

    env = data.pop("env")
    assert env["num_workers"] == "4"

    evaluation = data.pop("evaluation")
    assert evaluation["quantiles"] == [0.1, 0.5, 0.9]
