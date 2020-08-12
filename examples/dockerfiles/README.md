# Instructions for using the Dockerfiles

To build these dockerfiles go to the Gluon-TS root directory and enter this command:

```bash 
docker build . -f examples/dockerfiles/<dockerfile>
```

Or alternatively in the current directory:

```bash
docker build ../.. -f <dockerfile>
```

The built images are compatible with sagemaker.
For more information about the shell and the available params, see the [shell documentation](https://github.com/awslabs/gluon-ts/tree/master/src/gluonts/shell).


## How to choose between the dockerfiles

* Dockerfile
    - Gluon-TS models using CPU.
* Dockerfile.gpu
    -  Gluon-TS models on a GPU accelerated machine. 
* Dockerfile.r
    - This provides dependencies for models defined in [gluonts.model.r_forecast](https://gluon-ts.mxnet.io/api/gluonts/gluonts.model.r_forecast.html).
