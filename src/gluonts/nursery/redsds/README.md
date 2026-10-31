# Recurrent Explicit Duration Switching Dynamical Systems (RED-SDS)

[![Venue:NeurIPS 2021](https://img.shields.io/badge/Venue-NeurIPS%202021-007CFF)](https://neurips.cc)

This repository contains a reference implementation of RED-SDS, a non-linear state space model proposed in the NeurIPS 2021 paper *Deep Explicit Duration Switching Models for Time Series*. 

## Environment Setup

* Run `pip install -r requirements.txt`.

## Usage

### Reevaluating Trained Models

* Download the trained models from [this link](https://drive.google.com/drive/folders/19DmHwmsDZGbU4WEujwp0j1HWMdA5dDhU?usp=sharing).
* Run `python reevaluate.py --ckpt <model-path>.pt`.

### Training Models

#### Segmentation

* Generate/download datasets.
    * To generate the bouncing ball and 3 mode system datasets, use the notebooks in `./data/`. Alternatively, you can download the datasets from [this link](https://drive.google.com/drive/folders/1g5O2jktqWnH2p1BCkn1WtBBmSEF5AMTD?usp=sharing).
    * To download and preprocess the dancing bees dataset, run `./data/bee.sh`.
* Run `python run_segmentation.py --config configs/<config>.yaml --device cuda:0` to train the RED-SDS model.
* Run `tensorboard --logdir /path/to/results/dir` to visualize results.

#### Forecasting

* Run `python run_gts_univariate.py --config configs/<config>.yaml --device cuda:0` to train the RED-SDS model. The dataset will be downloaded automatically.
* Run `tensorboard --logdir /path/to/results/dir` to visualize results.

## Questions

For any questions regarding the code or the paper, please email [Fatir](mailto:abdulfatir@u.nus.edu), [Konstantinos](mailto:kbenidis@amazon.com), or [Richard](mailto:kurler@amazon.com).

## BibTeX

If you find this repository or the ideas presented in our paper useful for your research, please consider citing our paper.

```
@inproceedings{ansari2021deep,
  author    = {Abdul Fatir Ansari and Konstantinos Benidis and Richard Kurle and Ali Caner Turkmen and Harold Soh and Alex Smola and Bernie Wang and Tim Januschowski},
  title     = {Deep Explicit Duration Switching Models for Time Series},
  year      = {2021},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
}
```

## Acknowledgement

This repo contains parts of code based on the following repos:

| Repo  | Copyright (c) | License |
| ------------- | ---------- | ------------- |
| [google-research/google-research/snlds](https://github.com/google-research/google-research/tree/master/snlds)  | The Google Research Authors | [Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0) |
| [mattjj/pyslds](https://github.com/mattjj/pyslds) | Matthew James Johnson | [MIT](https://github.com/mattjj/pyslds/blob/master/LICENSE-MIT)