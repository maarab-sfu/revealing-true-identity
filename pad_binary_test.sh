#!/usr/bin/env bash
source ~/.bashrc
conda activate batl-utils
nvidia-smi
whereis cuda
cat /usr/local/cuda/version.txt
cd /nas/home/aminarab/batl-utils/
python driver.py --input_nc 6 --output_nc 6 --data custom --dataroot /nas/vista-ssd01/batl/SFU/Dataset2 --dataset_mode unaligned --continue_train --epoch best --niter 1 --niter_decay 0 --name ISI_binary --model binary --crop_size 32 --load_size 38 --batch_size 1 --lr 0.00000001 --infra_bands swir-1050nm#swir-1200nm#swir-1450nm --binary_test 1 --binary_thresh 0.0348

