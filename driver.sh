#!/usr/bin/env bash
source ~/.bashrc
conda activate batl-utils
nvidia-smi
whereis cuda
cat /usr/local/cuda/version.txt
cd /nas/home/aminarab/batl-utils/
python driver_extract.py 
