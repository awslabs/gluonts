# Instructions for using the Dockerfiles

To build these dockerfiles go to the Gluon-TS root directory and run:

```bash 
docker build . -f examples/dockerfiles/<the dockerfile>
```

To run the built image run:

```bash
docker run <image_id> <shell_params>
```

For more information about the shell and the available params, see the [shell documentation](https://github.com/awslabs/gluon-ts/tree/master/src/gluonts/shell) or run:

```bash
docker run <image_id> --help
```

## How to choose between the dockerfiles

* Dockerfile
    - This is for running Gluon-TS algorithms on a machine using only the CPU. This image is the most basic one and allows the use of most of the gluonts forecasters.
* Dockerfile.gpu
    - This is for running Gluon-TS algorithms on a GPU accelerated machine. 
* Dockerfile.r
    - This is used when one wants to run the models in the [gluonts.model.r_forecast](https://gluon-ts.mxnet.io/api/gluonts/gluonts.model.r_forecast.html) package.
