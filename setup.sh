#!/bin/bash

conda create -n nas python=3.9 -y
source activate nas #source not found err: run ./ not sh

pip install numpy
pip install tensorflow==2.12.0
pip install tflearn==0.5.0

pip install flask
pip install scikit-image

#install torch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
# conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia

pip install chardet
pip install opencv-python

pip uninstall -y Pillow #to match the version
pip install Pillow==9.5.0
pip install xlsxwriter

conda deactivate