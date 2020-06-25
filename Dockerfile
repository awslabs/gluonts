FROM python:3.7

RUN pip install mxnet==1.6

# ADD . /gluonts
# RUN pip install /gluonts[shell]

RUN pip install gluonts[shell]==0.5.

ENTRYPOINT ["python", "-m", "gluonts.shell"]
