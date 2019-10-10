#!/bin/bash
env_name=$1

export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export CUDA_VISIBLE_DEVICES=$EXECUTOR_NUMBER
export CONDA_ENVS_PATH=$PWD/conda
export CONDA_PKGS_DIRS=$PWD/conda/pkgs

make clean
conda env update --prune -p conda/${env_name} -f env/${env_name}.yml
conda activate ./conda/${env_name}
conda list
printenv

pip install -v -e ".[shell,Prophet]"
