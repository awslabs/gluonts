FROM python:3.11

ADD . /gluonts

RUN pip install "/gluonts[mxnet,shell]"

ENTRYPOINT ["python", "-m", "gluonts.shell"]