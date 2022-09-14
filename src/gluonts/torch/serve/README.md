# GluonTS TorchServe Handlers

GluonTS PyTorch models can be served using TorchServe.

## Packaging Models
In order to prepare a model for serving with TorchServe, first serialize the predictor with `predictor.serialize(file_name, use_torchscript=True)`.

You can then package the resulting serialzed predictor using the `torch-model-archiver`, e.g.:

    torch-model-archiver --model-name test -v 1 \
    --serialized-file test_model/prediction_net.pt \
    --handler /path/to/gluonts/src/gluonts/torch/serve/handler.py \
    --extra-files test_model/input_transform.json,test_model/parameters.json,test_model/type.txt,test_model/version.json

The resulting test.mar file can then be copied into the TorchServe model store.

You can then make inference requests to the API, e.g.

    curl -X POST \
    -H "Content-Type: application/json"  \
    http://localhost:8080/predictions/test \
    -d '{"target": [1, 2, 3], "start": "2020-01-01"}'

## Customization

You can customize the model output format by subclassing the GluonTSHandler and overriding the `postprocess()` method.