#!/usr/bin/env bash
source ~/.bashrc
conda activate batl-utils
nvidia-smi
whereis cuda
cat /usr/local/cuda/version.txt
cd /nas/home/aminarab/batl-utils/
python driver.py --input_nc 6 --output_nc 6 --niter 300  --niter_decay 500 --data custom --name test --crop_size 256 --load_size 256 --model cycle_gan --dataroot /nas/vista-ssd01/batl/SFU/Dataset2 --dataset_mode unaligned --netG resnet_6blocks+ --prob 0.075 --netD dilated  --ndf 32 --perceptual_factor 1.0 --lambda_class 0.0 --lambda_feature 2.0 --lambda_face_identity 0.02 --infra_bands swir-1050nm#swir-1200nm#swir-1450nm --stddev 0.1 
# python driver.py --input_nc 3 --output_nc 3 --batch_size 8 --niter 200 --niter_decay 600 --data custom --name pix2pix_shoe --model pix2pix --dataroot /nas/vista-ssd01/batl/SFU/pix2pix_dataset --dataset_mode aligned --prob 0.075 --netG unet_64 --perceptual_factor 1.0 --lambda_class 0.0 --lambda_feature 4.0 --lambda_face_identity 0.01
#python driver.py --input_nc 3 --output_nc 3 --batch_size 8 --direction AtoB --niter 200 --niter_decay 600 --dataroot /nas/home/aminarab/batl-utils/MU_dataset --name MU --infra_bands swir-1050nm#swir-1200nm#swir-1450nm --model cyclegan  --dataset_mode unaligned --prob 0.075 --netG unet_256

