FROM nvidia/cuda:9.2-cudnn7-runtime-ubuntu18.04

RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    wget bzip2 ca-certificates curl git build-essential \
    cmake python3.7 python3.7-dev python3.7-distutils pkg-config && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3.7 get-pip.py

# We're emulating nvidia-smi to be able to build GPU container on devices without GPU
RUN echo '#!/bin/bash\necho Building container with GPU support' > /usr/bin/nvidia-smi && \
    chmod +x /usr/bin/nvidia-smi

ADD . /gluonts

RUN pip3.7 install mxnet-cu92>=1.6.0
RUN pip3.7 install /gluonts[shell]
ENV MXNET_USE_FUSION=0

ENTRYPOINT ["python3.7", "-m", "gluonts.shell"]
