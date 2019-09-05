FROM python:3

ADD . /gluonts

RUN pip install /gluonts[shell]

ENTRYPOINT ["python", "-m", "gluonts.shell"]
