#!/usr/bin/env bash
source ~/.bashrc
conda activate batl-utils
nvidia-smi
whereis cuda
cat /usr/local/cuda/version.txt
cd /nas/home/aminarab/batl-utils/
python driver.py --input_nc 6 --output_nc 6 --attack_only --dataroot /nas/vista-ssd01/batl/SFU/Dataset2 --batch_size 1 --dataset_mode unaligned --data custom --name ISI_nopre --model cycle_gan --epoch best --netG resnet_6blocks+ --netD dilated --mode test --mask diff --method rgb --matcher Verilook --binary_thresh 0.50 --crop_size 256 --load_size 256 --infra_bands swir-1050nm#swir-1200nm#swir-1450nm
