# Time series embeddings

This code is a modification of the  [code](https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries) from the paper: 
[_Unsupervised Scalable Representation Learning for Multivariate Time Series_](https://arxiv.org/abs/1901.10738).
The main difference is that this code uses the NT-Xent loss as in SimCLR.

Example for training an encoder
```sh
python run.py --dataset_name=traffic --ts_len=14000 --compared_length=500 --lr=0.005 --loss_temperature=0.2 --batch_size=256 --max_epochs=20 --multivar_dim=1
```
This trains an encoder, which is then serialized as `encoder.pt`. If you choose a dataset that is 
not contained in the repository, then it should be following the GluonTS conventions of a `FileDataset`.

You can then run the `ShowEmbeddings.ipynb` notebook to visualize embeddings and time series.

Note: you may need to set `--num_workers=0` according to
`https://github.com/pytorch/pytorch/issues/46409`
if you run into problems.

In the multi-variate setting, it is assumed that the each multi-variate time series has the same dimension k and these
are consecutive and in the same order, e.g., a tensor of size `( n times k, T)` where `T` is the number of time steps, 
`N` is the number of `k`-variate time series. This is then transformed into a tensor of shape `(n, k, T)`.