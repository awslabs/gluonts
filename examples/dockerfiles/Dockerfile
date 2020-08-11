FROM python:3.7

RUN apt-get update && apt-get install -y r-base
RUN R -e "install.packages('forecast',dependencies=TRUE, repos='http://cran.rstudio.com/')"
RUN R -e "install.packages('nnfor',dependencies=TRUE, repos='http://cran.rstudio.com/')"

ADD . /gluonts

RUN pip install -r /gluonts/requirements/requirements-mxnet.txt
RUN pip install /gluonts[shell,R]

ENTRYPOINT ["python", "-m", "gluonts.shell"]
