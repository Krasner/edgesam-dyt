#!/bin/bash

sudo apt install pybind11-dev
sudo apt install python3-pybind11

cd ~
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
pip install .

cd ~
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install .

cd ~
git clone https://github.com/open-mmlab/mmengine.git
cd mmengine
pip install .