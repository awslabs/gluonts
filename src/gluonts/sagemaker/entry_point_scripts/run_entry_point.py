import argparse
import os
import json


# TODO make it work with any gluonts algorithm: argument to pass GluonEstimator
# TODO make it work with custom algorithms defined in the source dir
# TODO add option to allow for continuing for training
def run(arguments):

    # print arguments
    print(str(arguments))

    # load the file
    # / opt / ml / input / data / testing
    with open("/opt/ml/input/data/testing/dataset.txt", "rt") as f:
        text = f.read()
        print(f"File text of length: {len(text)}")

    print("done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # # hyperparameters sent by the client are passed as command-line arguments to the script.
    # parser.add_argument('--epochs', type=int, default=10)
    # parser.add_argument('--batch-size', type=int, default=100)
    # parser.add_argument('--learning-rate', type=float, default=0.1)

    # an alternative way to load hyperparameters via SM_HPS environment variable.
    parser.add_argument('--sm-hps', type=json.loads, default=os.environ['SM_HPS'])

    # input data and model directories
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])  # specified in
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])  # specified in fit: inputs
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])  # specified in fit: inputs

    args, _ = parser.parse_known_args()

    # ... load from args.train and args.test, train a model, write model to args.model_dir.
    run(args)
