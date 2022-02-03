#!/bin/bash
set -e

#--------------------------------------------------------------------------------------------------
echo "Installing packages..."
sudo apt-get update
sudo apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
    libsqlite3-dev wget curl llvm libncurses5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
    libffi-dev liblzma-dev libcurl4-openssl-dev r-base p7zip-full awscli

#--------------------------------------------------------------------------------------------------
echo "Installing Node.js..."
curl -fsL https://deb.nodesource.com/setup_16.x | sudo bash
sudo apt-get install -y nodejs

#--------------------------------------------------------------------------------------------------
echo "Installing R forecast..."
sudo R -e 'install.packages(c("forecast", "nnfor"), repos="https://cloud.r-project.org")'

#--------------------------------------------------------------------------------------------------
echo "Installing Python..."
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
cd ~/.pyenv && src/configure && make -C src
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.profile
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.profile
echo 'eval "$(pyenv init --path)"' >> ~/.profile
source ~/.profile
pyenv install 3.8.9
echo "3.8.9" > ~/.python-version

#--------------------------------------------------------------------------------------------------
echo "Installing Poetry..."
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
source $HOME/.poetry/env
poetry config virtualenvs.in-project true

#--------------------------------------------------------------------------------------------------
echo "Installing Docker..."
sudo apt-get update
sudo apt-get install -y apt-transport-https ca-certificates curl gnupg lsb-release
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
    sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo \
  "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io
sudo usermod -aG docker $USER

#--------------------------------------------------------------------------------------------------
echo "Installing MongoDB..."
wget -qO - https://www.mongodb.org/static/pgp/server-5.0.asc | sudo apt-key add -
echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/5.0 multiverse" | \
    sudo tee /etc/apt/sources.list.d/mongodb-org-5.0.list
sudo apt-get update
sudo apt-get install -y mongodb-org
sudo systemctl restart mongod
