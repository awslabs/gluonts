FROM python:3.8.12-buster

# Install R
RUN apt-get update \
    && apt-get install -y r-base \
    && R -e 'install.packages(c("forecast", "nnfor"), repos="https://cloud.r-project.org")'

# Install project dependencies
RUN pip install poetry==1.1.6 \
    && poetry config virtualenvs.create false
COPY poetry.lock pyproject.toml /dependencies/
RUN cd /dependencies \
    && poetry install --no-dev --no-root --no-interaction --no-ansi
