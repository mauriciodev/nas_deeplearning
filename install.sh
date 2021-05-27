#!/bin/bash

conda create -p `pwd`/autodeeplab -c conda-forge
conda activate `pwd`/autodeeplab
conda activate "autodeeplab"
conda install -c conda-forge tensorflow-gpu=="1.15" pillow tf_slim jupyterlab


conda remove --force cudnn
conda remove --force cudatoolkit

echo "Cudnn and cuda toolkit should be installed from NVIDIA"
