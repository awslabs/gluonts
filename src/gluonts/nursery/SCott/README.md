# SCott

This repo contains the experimental code for paper

[Variance Reduced Training with Stratified Sampling for Forecasting Models](https://arxiv.org/pdf/2103.02062.pdf), in [ICML'21](https://icml.cc/Conferences/2021)

[Yucheng Lu](https://www.cs.cornell.edu/~yucheng/), [Youngsuk Park](https://youngsuk0723.github.io/), [Lifan Chen](https://www.amazon.science/author/lifan-chen), [Yuyang (Bernie) Wang](http://www.mit.edu/~ywang02/), [Christopher De Sa](http://www.cs.cornell.edu/~cdesa/), [Dean Foster](http://deanfoster.net/)

Our implementation is based on the open source framework: [PytorchTS](https://github.com/zalandoresearch/pytorch-ts). If you would like to cite our paper, please use the following bibtex:

```
@inproceedings{lu2021variance,
  title={Variance Reduced Training with Stratified Sampling for Forecasting Models},
  author={Lu, Yucheng and Park, Youngsuk and Chen, Lifan and Wang, Yuyang and De Sa, Christopher and Foster, Dean},
  booktitle={International Conference on Machine Learning},
  pages={7145--7155},
  year={2021},
  organization={PMLR}
}
```

## (Preparation Step 1) Installation of required packages

The details of the required packages can be found in `requirements.txt`.

The following example commands offer ways of installing required packages with `conda`. One can use other softwares such as `pip` to do this.

```
$ conda install pytorchts==0.2.0
```

## (Preparation Step 2) Preprocessing the dataset

After installing the packages, we should generate the dataset with

```
$ python3 preprocess_data.py
```

which should produce four `.csv` files in the `./dataset` repo.

## Commands for each experiment

Please refer to the [commands](https://github.com/lovvge/SCott/tree/main/commands) repo.

## Visualize the results

One can check the results in the `./runs` repo, where the results are in the tensorboard formats. To check these results, we can simply run

```
tensorboard --logdir ./runs --port <port>
```

and open a local browser and go to `localhost:<port>`. If the experiments are done on a remote host, we should add one more step by

```
ssh -NfL <port>:localhost:<port> hostname
```

For more info regarding usage of tensorboard, please refer to [torch.utils.tensorboard](https://pytorch.org/docs/stable/tensorboard.html).

## Errata (Updated 08/2021)
* The learning rate for Adam/SAdam on N-BEATS should be 1e-3 instead of 1e-2.
