FROM ubuntu:16.04

LABEL maintainer="Amazon ML & Forecasting"

LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true

ARG MX_VERSION=1.4.1
ARG PYTHON=python3
ARG PYTHON_PIP=python3-pip
ARG PIP=pip3
ARG PYTHON_VERSION=3.6.6

ARG REQUIREMENTS=requirements.txt
ARG REQUIREMENTS_EXTRAS_R=requirements-extras-r.txt
ARG REQUIREMENTS_EXTRAS_PROPHET=requirements-extras-prophet.txt

RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        wget \
        git \
        libopencv-dev \
        openssh-client \
        openssh-server \
        vim \
        zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

# Install R
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9 && \
    add-apt-repository 'deb [arch=amd64,i386] http://cran.rstudio.com/bin/linux/ubuntu xenial/' && \
    apt-get update && \
    apt-get install -y r-base r-base-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Python
RUN wget https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz && \
        tar -xvf Python-$PYTHON_VERSION.tgz && cd Python-$PYTHON_VERSION && \
        ./configure && make && make install && \
        apt-get update && apt-get install -y --no-install-recommends libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev && \
        make && make install && rm -rf ../Python-$PYTHON_VERSION* && \
        ln -s /usr/local/bin/pip3 /usr/bin/pip

RUN ${PIP} --no-cache-dir install --upgrade pip setuptools
RUN ln -s $(which ${PYTHON}) /usr/local/bin/python

WORKDIR /

COPY sagemaker_mxnet_container.tar.gz /sagemaker_mxnet_container.tar.gz

# Install MXNet and current sagemaker-mxnet-containers version
RUN ${PIP} install --no-cache --upgrade mxnet-mkl==$MX_VERSION && \
    ${PIP} install --no-cache-dir /sagemaker_mxnet_container.tar.gz && \
    rm /sagemaker_mxnet_container.tar.gz

# Add other GluonTS requirements
ADD \
    $REQUIREMENTS \
    $REQUIREMENTS_EXTRAS_R \
    $REQUIREMENTS_EXTRAS_PROPHET \
    /tmp/dependencies/

# Install GluonTS requirements, and current GluonTS version
RUN ${PIP} install --no-cache --upgrade -r /tmp/dependencies/requirements.txt && \
    ${PIP} install --no-cache --upgrade -r /tmp/dependencies/requirements-extras-r.txt && \
    ${PIP} install --no-cache --upgrade -r /tmp/dependencies/requirements-extras-prophet.txt && \
    ${PIP} install --no-cache --upgrade gluonts

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib" \
    PYTHONIOENCODING=UTF-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

ENV SAGEMAKER_TRAINING_MODULE sagemaker_mxnet_container.training:main

CMD ["/bin/bash"]
