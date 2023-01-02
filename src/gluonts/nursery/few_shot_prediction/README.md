# Probabilistic Few-Shot Forecasting
We propose a method for probabilistic few-shot time series forecasting.
In this setting, the model is trained on a large and diverse collection of source data sets and can subsequently make probabilistic predictions for previously unseen target data sets given a small number of representative examples (the support set), without additional retraining or fine-tuning.
Performance can be improved by choosing the examples in the support set based on a notion of distance to the target time series during both training and prediction.
The codebase allows training and evaluating our method on a large collection of publicly available real-world data sets.

### Set up

```bash
# install pyenv
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
cd ~/.pyenv && src/configure && make -C src
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.profile
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.profile
echo 'eval "$(pyenv init --path)"' >> ~/.profile
source ~/.profile
pyenv install 3.8.9

# install poetry
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
source $HOME/.poetry/env
poetry config virtualenvs.in-project true

# create virtual env for your project
cd to_your_project
pyenv local 3.8.9
poetry config virtualenvs.in-project true
poetry install
```

### Generate and upload data
Run `scripts/data.py real` to download and process the GluonTS datasets supported be the few-shot predictor. 

### Train a model
Run `scripts/train.py` with specific arguments to start a training.



