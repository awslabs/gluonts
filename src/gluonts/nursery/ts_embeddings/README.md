# Time series embeddings

This code is a modification of the  [code](https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries) from the paper: 
[_Unsupervised Scalable Representation Learning for Multivariate Time Series_](https://arxiv.org/abs/1901.10738).
The main difference is that this code uses the NT-Xent loss as in SimCLR.

Example for training an encoder
```sh
./run.py --dataset_name=traffic --ts_len=14000 --compared_length=500 --lr=0.005 --loss_temperature=0.2 --batch_size=256 --max_epochs=20
```

This trains an encoder, which is then serialized as `encoder.pt`.

You can then run the `ShowEmbeddings.ipynb` notebook to visualize embeddings and time series.
