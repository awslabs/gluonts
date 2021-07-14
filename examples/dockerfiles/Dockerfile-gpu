FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/mxnet-training:1.6.0-gpu-py36-cu101-ubuntu16.04

ADD . /gluonts

#RUN pip install -r /gluonts/requirements/requirements-mxnet-gpu.txt
RUN pip install /gluonts[shell]

ENTRYPOINT ["python", "-m", "gluonts.shell"]
