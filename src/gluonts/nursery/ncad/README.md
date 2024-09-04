# NCAD - Neural Contextual Anomaly Detection for Time Series

[![GitHub](https://img.shields.io/:license-apache-blue.svg?style=flat-square)](./LICENSE)

NCAD is a framework for anomaly detection in time series that achieves state-of-the-art
performance in both the univariate and multivariate cases, as well as from the unsupervised
to the supervised regime.


## Citing

If you use NCAD in a scientific publication, we encourage you to add
the following references to the related papers:

```bibtex
@article{Carmona2021ncad,
    author = {Carmona, Chris and Aubet, Fran\c{c}ois-Xavier and Flunkert, Valentin and Gasthaus, Jan},
    title = {Neural Contrastive Anomaly Detection},
    year = {2021}
    url = {https://github.com/awslabs/gluon-ts/tree/master/src/gluonts/nursery/ncad},
}
```

## Installation instructions

1. Clone GluonTS github repository
```
git clone https://github.com/awslabs/gluon-ts.git
```
2. \[Optional] Create a new virtual environment for this project (see [*Create a virtual environment*](#create-a-virtual-environment) below).
3. Install `ncad` module
```
pip install -e gluon-ts/src/gluonts/nursery/ncad
```
4. \[Optional] Download benchmark datasets (see [*data*](#data) section below).


## Examples
All the benchmark results in our Neurips2021 article can be reproduced. We have included hyperparameter configuration files for each dataset, as well as a python script that sequentially train the model for each one.
Hyperparameter files can be found under `examples/article/hparams`.
To train these models sequentially, go the the main repository directory and run the following command (adjusting for your directories):
```bash
python3 examples/article/run_all_experiments.py \
--ncad_dir='~/ncad' \
--data_dir='~/ncad_datasets' \
--hparams_dir='~/ncad/examples/article/hparams' \
--out_dir='~/ncad_output' \
--download_data=True \
--number_of_trials=10 \
--run_swat=False \
--run_yahoo=False
```

This will automatically download all datasets that can be reached online and train one model for each hyperparameter file.
Note that in the code above we have set the options `run_swat=False` and `run_yahoo=False`, this due to the fact that these datasets must be requested and can't be directly downloaded (see [*data*](#data) section below).

If you have obtained the `yahoo` and `SWaT` datasets and want to run all experiments, simply run
```bash
python3 examples/article/run_all_experiments.py \
--ncad_dir='~/ncad' \
--data_dir='~/ncad_datasets' \
--hparams_dir='~/ncad/examples/article/hparams' \
--out_dir='~/ncad_output' \
--download_data=True \
--yahoo_path='~/ncad_datasets/yahoo_dataset.tgz' \
--number_of_trials=10
```


## Data

Examples in this repository depend on the following benchmark datasets:
1. KPI competition data.
    More detals (in chinese) on the dataset and competition are provided at their [website](http://iops.ai/problem_detail/?id=5).
    Data available in this [Github repository](https://github.com/NetManAIOps/KPI-Anomaly-Detection)
2. NASA's SMAP and MSL.
    More details about these datasets provided in the Telemanom [Github repository](https://github.com/khundman/telemanom).
3. Server Machine Dataset (SMD).
    More details about theis datasets provided in the OmniAnomaly [Github repository](https://github.com/NetManAIOps/OmniAnomaly).
4. Secure Water Treatment (SWaT) data
    The data is available upon request to iTrust Labs via their [website](https://itrust.sutd.edu.sg/itrust-labs_datasets/)
5. Yahoo webscope.
    The data is available upon request to Yahoo Labs: [S5 - A Labeled Anomaly Detection Dataset, version 1.0](https://webscope.sandbox.yahoo.com/catalog.php?datatype=s&did=70)

We have included a function to download the benchmark datasets. `kpi`,`nasa` and `smd` can be downloaded automatically, but `swat` and `yahoo` require users' request.
```python
from ncad.datasets import download
download(
    data_dir = "~/ncad_datasets",
    benchmarks = ['kpi','nasa','smd'],
)
```
For SWaT, you need to manually create the `swat` directory and save the `Dec2015` files provided by iTrust Labs.
For Yahoo, If you have obtained the `.tgz` file from Yahoo Labs, you can pass its location and the function will automatically uncompress it and include it in the data directory.
```python
download(
    data_dir = "~/ncad_datasets",
    benchmarks = ['kpi','nasa','smd','yahoo'],
    yahoo_path = "~/ncad_datasets/yahoo_dataset.tgz",
)
```

A directory with all the datasets should have the following structure:
```bash
~/ncad_datasets
├── kpi
│   ├── phase2_ground_truth.hdf
│   └── phase2_train.csv
├── nasa
│   ├── labeled_anomalies.csv
│   ├── test
│   │   ├── A-1.npy
│   │   │   ...
│   │   └── T-9.npy
│   └── train
│       ├── A-1.npy
│       │   ...
│       └── T-9.npy
├── smd
│   ├── interpretation_label
│   │   ├── machine-1-1.txt
│   │   ├── ...
│   │   └── machine-3-9.txt
│   ├── test
│   │   ├── machine-1-1.txt
│   │   ├── ...
│   │   └── machine-3-9.txt
│   ├── test_label
│   │   ├── machine-1-1.txt
│   │   ├── ...
│   │   └── machine-3-9.txt
│   └── train
│       ├── machine-1-1.txt
│       ├── ...
│       └── machine-3-9.txt
├── swat
│   ├── SWaT_Dataset_Attack_v0.csv
│   └── SWaT_Dataset_Normal_v0.csv
└── yahoo
    ├── A1Benchmark
    │   ├── real_1.csv
    │   ├── ...
    │   └── real_9.csv
    ├── A2Benchmark
    │   ├── synthetic_1.csv
    │   ├── ...
    │   └── synthetic_99.csv
    ├── A3Benchmark
    │   ├── A3Benchmark-TS1.csv
    │   ├── ...
    │   └── A3Benchmark_all.csv
    └── A4Benchmark
        ├── A4Benchmark-TS1.csv
        ├── ...
        └── A4Benchmark_all.csv

21 directories, 660 files
```



## Contributors

Thank you to the following for their contributions to this project:

- [Chris U Carmona](https://chriscarmona.me)
- [François-Xavier Aubet](mailto:aubetf@amazon.com)
- [Jan Gasthaus](mailto:gasthaus@amazon.com)
- [Valentin Flunkert](mailto:flunkert@amazon.com)


### Create a virtual environment

For OSX or Linux, you can use `venv` (see the [venv documentation](https://docs.python.org/3/library/venv.html)).

Create `ncad` virtual environment
```bash
python3 -m venv ~/.virtualenvs/ncad
```
Activate the virtual environment
```bash
source ~/.virtualenvs/ncad/bin/activate
```
Upgrade pip
```bash
pip install -U pip
```

Feel free to change the folder where the virtual environment is created by replacing `~/.virtualenvs/ncad` with a path of your choice.
