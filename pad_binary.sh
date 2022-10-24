#!/usr/bin/env bash
source ~/.bashrc
conda activate batl-utils
nvidia-smi
whereis cuda
cat /usr/local/cuda/version.txt
cd /nas/home/aminarab/batl-utils/
#python driver.py --input_nc 3 --dataset_mode unaligned --dataroot /nas/vista-ssd01/batl/SFU/Makeup_Dataset --data custom --niter 100 --niter_decay 200 --name makeup_binary_rgb_3channels --model binary --crop_size 32 --load_size 38 --batch_size 8 --lr 0.0001  
#cp /nas/vista-ssd01/batl/SFU/results/checkpoints/finetune_rgb_makeup/best* /nas/home/aminarab/batl-utils/checkpoints/st4_binary
#python driver.py --input_nc 6 --dataset_mode h5 --data $d --niter 100 --niter_decay 200 --name makeup_binary_$d-$b --continue_train --epoch best --model binary --crop_size 32 --load_size 38 --batch_size 8 --lr 0.0001 --infra_bands $b
#python driver.py --input_nc 6 --output_nc 6 --data custom --dataroot /nas/home/aminarab/batl-utils/dataset --dataset_mode unaligned --niter 200 --niter_decay 200 --name all_binary_new --model binary --crop_size 32 --load_size 38 --batch_size 1 --lr 0.0001 --infra_bands swir-1050nm#swir-1200nm#swir-1450nm
python driver.py --input_nc 6 --output_nc 6 --data custom --dataroot /nas/vista-ssd01/batl/SFU/Dataset2 --dataset_mode unaligned --niter 300 --niter_decay 300 --name ISI_binary --model binary --crop_size 32 --load_size 38 --batch_size 1 --lr 0.0001 --infra_bands swir-1050nm#swir-1200nm#swir-1450nm
#python driver.py --input_nc 6 --output_nc 6 --data custom --dataroot /nas/vista-ssd01/batl/SFU/Dataset --dataset_mode unaligned --niter 1 --niter_decay 0 --name test --model binary --crop_size 32 --load_size 38 --batch_size 1 --lr 0.00000001 --infra_bands swir-1050nm#swir-1200nm#swir-1450nm

