import argparse
import os
import json
import time


def run(arguments):
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "Starting - started running custom script.")
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # load hyperparameters via SM_HPS environment variable
    parser.add_argument('--sm_hps', type=json.loads, default=os.environ['SM_HPS'])

    # input data, output dir and model directories
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--input_dir', type=str, default=os.environ['SM_INPUT_DIR'])
    # contents will be written back to s3
    parser.add_argument('--output_data_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])

    args, _ = parser.parse_known_args()

    run(args)
