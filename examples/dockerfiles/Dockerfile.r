FROM rpy2/base-ubuntu:latest
# Change the tag in the image_name:tag above to a different version, if required.
# Available tags: https://hub.docker.com/r/rpy2/base-ubuntu/tags

ADD . /gluonts

RUN pip install "/gluonts[shell,R]"

RUN R -e 'install.packages(c("forecast", "hts"), repos="https://cloud.r-project.org")'
RUN R -e 'install.packages("nnfor", repos="https://cloud.r-project.org")'

ENV LC_ALL="C.UTF-8"
ENV LANG="C.UTF-8"

ENTRYPOINT ["python3", "-m", "gluonts.shell"]