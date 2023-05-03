# Robust Multivariate Time-Series Forecasting: Adversarial Attacks and Defense Mechanisms

This repository contains the source code for reproducing the experiment results in the paper 

**Robust Multivariate Time-Series Forecasting: Adversarial Attacks and Defense Mechanisms, ICLR 2023.**<br>
*Linbo Liu, Youngsuk Park, Trong Nghia Hoang, Hilaf Hasson, Jun Huan.*

## Requirements
PyTorch (1.13.1)<br>
gluonts (0.9.2) (available at https://github.com/mbohlkeschneider/gluon-ts/tree/mv_release)<br>
pytorch-ts from [Zalando Research](https://github.com/zalandoresearch/pytorch-ts)

## Datasets
The datasets (Traffic, Taxi, Wiki, Electricity) can be downloaded at https://github.com/awslabs/gluonts/blob/dev/src/gluonts/dataset/repository/_gp_copula_2019.py

## Example
We provide an example to train, attack and evaluate our model on Traffic dataset.

#### Train
* To train a clean model, run
```
python train.py trained_models/traffic/dim10-rank5/ --max_target_dim 10 --rank 5 --epochs 50 --dataset traffic --prediction_length 24
```
* To train a model with data augmentation, run
```
python train.py trained_models/gaussian/traffic/sigma_0.1/dim10-rank5/ --max_target_dim 10 --rank 5 --epochs 50 --dataset traffic --prediction_length 24 --gaussian
```
* To train a model with mini-max defense, run
```
python train_adv.py trained_models_adv/traffic/dim10-rank5-s5/ --max_target_dim 10 --rank 5 --epochs 50 --dataset traffic --prediction_length 24 --sparsity 5 --attack_params attack_params/attack_config_traffic_4.json --lr 0.0001
```

#### Generate deterministic adversarial attack
* For clean model, run
```
python attack_and_save.py trained_models/traffic/dim10-rank5/ --dataset traffic --attack_params attack_params/attack_config_traffic_4.json
```
* For model with data augmentation, run
```
python attack_and_save.py trained_models/gaussian/traffic/sigma_0.1/dim10-rank5/ --dataset traffic --attack_params attack_params/attack_config_traffic_4.json
```
* For model with mini-max defense, run
```
python attack_and_save.py trained_models_adv/traffic/dim10-rank5-s5/ --dataset traffic --attack_params attack_params/attack_config_traffic_4.json
```
#### Evaluation
* For clean model, run
```
python eval.py trained_models/traffic/dim10-rank5/ --attack_params_path attack_params/attack_config_traffic_4.json --attack_result vanilla_24_traffic_sigma_0_rank5_dim10_4.pkl --dataset traffic
```
* For data augmentation, run
```
python eval.py trained_models/gaussian/traffic/sigma_0.1/dim10-rank5/ --attack_params_path attack_params/attack_config_traffic_4.json --attack_result rt_24_traffic_sigma_0.1_rank5_dim10_4.pkl --dataset traffic
```
* For randomized smoothing, run
```
python eval.py trained_models/gaussian/traffic/sigma_0.1/dim10-rank5/ --attack_params_path attack_params/attack_config_traffic_4.json --attack_result rt_24_traffic_sigma_0.1_rank5_dim10_4.pkl --dataset traffic --rs
```
* For mini-max, run
```
python eval.py trained_models_adv/traffic/dim10-rank5-s5/ --attack_params_path attack_params/attack_config_traffic_4.json --attack_result adv_24_traffic_sigma_0_rank5_dim10_s5_4.pkl --dataset traffic
```
#### Generate and evaluate probabilistic attack
* For clean model, run
```
python attack_sparse_layer.py trained_models/traffic/dim10-rank5/ --dataset traffic --attack_params attack_params/attack_config_traffic_4.json

python eval_sparse.py trained_models/traffic/dim10-rank5/ --attack_params_path attack_params/attack_config_traffic_4.json --attack_result vanilla_24_traffic_rank5_dim10_4.pkl --dataset traffic
```
Others have similar implementation

## Authors

Linbo Liu<br>
Youngsuk Park<br>
Trong Nghia Hoang<br>
Hilaf Hasson<br>
Jun Huan<br>




