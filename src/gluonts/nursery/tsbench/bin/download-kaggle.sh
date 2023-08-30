#!/bin/bash
set -e

# M5
echo "Downloading M5..."
poetry run kaggle competitions download -c m5-forecasting-accuracy -p ~/.mxnet/gluon-ts/datasets/
unzip ~/.mxnet/gluon-ts/datasets/m5-forecasting-accuracy.zip -d ~/.mxnet/gluon-ts/datasets/m5
rm -f ~/.mxnet/gluon-ts/datasets/m5-forecasting-accuracy.zip

# Rossmann
echo "Downloading Rossmann..."
poetry run kaggle competitions download -c rossmann-store-sales -p ~/.mxnet/gluon-ts/datasets/
unzip ~/.mxnet/gluon-ts/datasets/rossmann-store-sales.zip -d ~/.mxnet/gluon-ts/datasets/rossmann
rm -f ~/.mxnet/gluon-ts/datasets/rossmann-store-sales.zip

# Corporacion Favorita
echo "Downloading Corporacion Favorita..."
poetry run kaggle competitions download -c favorita-grocery-sales-forecasting \
    -p ~/.mxnet/gluon-ts/datasets/
unzip ~/.mxnet/gluon-ts/datasets/favorita-grocery-sales-forecasting.zip \
    -d ~/.mxnet/gluon-ts/datasets/corporacion_favorita
rm -f ~/.mxnet/gluon-ts/datasets/favorita-grocery-sales-forecasting.zip
ls ~/.mxnet/gluon-ts/datasets/corporacion_favorita | xargs -L1 7za e
rm -f ~/.mxnet/gluon-ts/datasets/corporacion_favorita/*.7z

# Restaurant
echo "Downloading Walmart..."
poetry run kaggle competitions download -c walmart-recruiting-store-sales-forecasting \
    -p ~/.mxnet/gluon-ts/datasets/
unzip ~/.mxnet/gluon-ts/datasets/walmart-recruiting-store-sales-forecasting.zip \
    -d ~/.mxnet/gluon-ts/datasets/walmart
rm -f ~/.mxnet/gluon-ts/datasets/walmart-recruiting-store-sales-forecasting.zip
find ~/.mxnet/gluon-ts/datasets/walmart -name "*.zip" | \
    xargs -L1 -I{} unzip {} -d ~/.mxnet/gluon-ts/datasets/walmart
rm -f ~/.mxnet/gluon-ts/datasets/walmart/*.zip

# Walmart
echo "Downloading Restaurant..."
poetry run kaggle competitions download -c recruit-restaurant-visitor-forecasting \
    -p ~/.mxnet/gluon-ts/datasets/
unzip ~/.mxnet/gluon-ts/datasets/recruit-restaurant-visitor-forecasting.zip \
    -d ~/.mxnet/gluon-ts/datasets/restaurant
rm -f ~/.mxnet/gluon-ts/datasets/recruit-restaurant-visitor-forecasting.zip
find ~/.mxnet/gluon-ts/datasets/restaurant -name "*.zip" | \
    xargs -L1 -I{} unzip {} -d ~/.mxnet/gluon-ts/datasets/restaurant
rm -f ~/.mxnet/gluon-ts/datasets/restaurant/*.zip
