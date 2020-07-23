FROM python:3.7

ADD . /gluonts

RUN pip install mxnet==1.6
RUN pip install /gluonts[shell]

ENTRYPOINT ["python", "-m", "gluonts.shell"]
