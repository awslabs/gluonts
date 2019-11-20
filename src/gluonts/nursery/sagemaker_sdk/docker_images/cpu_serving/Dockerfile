FROM ubuntu:16.04

LABEL maintainer="Amazon ML & Forecasting"

LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true

ARG MMS_VERSION=1.0.5
ARG MX_VERSION=1.4.1
ARG PYTHON=python3
ARG PYTHON_PIP=python3-pip
ARG PIP=pip3
ARG PYTHON_VERSION=3.6.8

ARG REQUIREMENTS=requirements.txt
ARG REQUIREMENTS_EXTRAS_R=requirements-extras-r.txt
ARG REQUIREMENTS_EXTRAS_PROPHET=requirements-extras-prophet.txt

RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && apt-get install -y --no-install-recommends \
    	build-essential \
    	ca-certificates \
    	curl \
    	git \
    	libopencv-dev \
    	openjdk-8-jdk-headless \
    	vim \
    	wget \
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

RUN ln -s $(which ${PYTHON}) /usr/local/bin/python

RUN ${PIP} --no-cache-dir install --upgrade pip setuptools

WORKDIR /

COPY sagemaker_mxnet_serving_container.tar.gz /sagemaker_mxnet_serving_container.tar.gz

# Install MXNet and current sagemaker-mxnet-containers version
RUN ${PIP} install --no-cache-dir --upgrade mxnet-mkl==$MX_VERSION && \
    ${PIP} install --no-cache-dir --upgrade mxnet-model-server==$MMS_VERSION && \
    ${PIP} install --no-cache-dir /sagemaker_mxnet_serving_container.tar.gz && \
    rm /sagemaker_mxnet_serving_container.tar.gz

# Add other GluonTS requirements
ADD \
    $REQUIREMENTS \
    $REQUIREMENTS_EXTRAS_R \
    $REQUIREMENTS_EXTRAS_PROPHET \
    /tmp/dependencies/

# Install GluonTS requirements, and current GluonTS version
RUN ${PIP} install --no-cache-dir --upgrade -r /tmp/dependencies/requirements.txt && \
    ${PIP} install --no-cache-dir --upgrade -r /tmp/dependencies/requirements-extras-r.txt && \
    ${PIP} install --no-cache-dir --upgrade -r /tmp/dependencies/requirements-extras-prophet.txt && \
    ${PIP} install --no-cache-dir --upgrade gluonts

# This is here to make our installed version of OpenCV work.
# https://stackoverflow.com/questions/29274638/opencv-libdc1394-error-failed-to-initialize-libdc1394
# TODO: Should we be installing OpenCV in our image like this? Is there another way we can fix this?
RUN ln -s /dev/null /dev/raw1394

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib" \
    PYTHONIOENCODING=UTF-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

RUN useradd -m model-server \
    && mkdir -p /home/model-server/tmp \
    && chown -R model-server /home/model-server

COPY mms-entrypoint.py /usr/local/bin/dockerd-entrypoint.py
COPY config.properties /home/model-server

RUN chmod +x /usr/local/bin/dockerd-entrypoint.py

EXPOSE 8080 8081
ENV TEMP=/home/model-server/tmp
ENTRYPOINT ["python", "/usr/local/bin/dockerd-entrypoint.py"]
CMD ["mxnet-model-server", "--start", "--mms-config", "/home/model-server/config.properties"]