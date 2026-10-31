wget -c https://www.cc.gatech.edu/~borg/ijcv_psslds/psslds.zip
unzip psslds.zip
rm psslds.zip
python preprocess_bee.py
rm -rf ./psslds
rm -rf ./__MACOSX