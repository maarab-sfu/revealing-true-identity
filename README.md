# revealing-true-identity
## Revealing True Identity: Detecting Makeup Attacks in Face-based Biometric Systems

Face-based authentication systems are among the most commonly used biometric systems, because of the ease of capturing face images at a distance and in non-intrusive way. These systems are, however, susceptible to various presentation attacks, including printed faces, artificial masks, and makeup attacks. In this paper, we propose a novel solution to address makeup attacks, which are the hardest to detect in such systems because makeup can substantially alter the facial features of a person, including making them appear older/younger by adding/hiding wrinkles, modifying the shape of eyebrows, beard, and moustache, and changing the color of lips and cheeks. In our solution, we design a generative adversarial network for removing the makeup from face images while retaining their essential facial features and then compare the face images before and after removing makeup. We collect a large dataset of various types of makeup, especially malicious makeup that can be used to break into remote unattended security systems. This dataset is quite different from existing makeup datasets that mostly focus on cosmetic aspects. We conduct an extensive experimental study to evaluate our method and compare it against the state-of-the art using standard objective metrics commonly used in biometric systems as well as subjective metrics collected through a user study. Our results show that the proposed solution produces high accuracy and substantially outperforms the closest works in the literature.

## Guide for makeup removal code

-	We train the classifier and the GAN separately. You can find bash scripts regarding each part.
- The structure of bash scripts is as follows:
```
#!/usr/bin/env bash
source ~/.bashrc
conda activate batl-utils
nvidia-smi
whereis cuda
cat /usr/local/cuda/version.txt
cd /nas/home/aminarab/batl-utils/
python driver.py --input_nc 6 --output_nc 6 --niter 300  --niter_decay 500 --data custom --name test --crop_size 256 --load_size 256 --model cycle_gan --dataroot /nas/vista-ssd01/batl/SFU/Dataset2 --dataset_mode unaligned --netG resnet_6blocks+ --prob 0.075 --netD dilated  --ndf 32 --perceptual_factor 1.0 --lambda_class 0.0 --lambda_feature 2.0 --lambda_face_identity 0.02 --infra_bands swir-1050nm#swir-1200nm#swir-1450nm --stddev 0.1 
```
- The arguments are pretty straight-forward and details about them can be found in the driver.py. (For example input_nc is the number of input channels, “output_nc” is the number of output channels, and etc.)
### How to run the code
-	First you need to setup the environment. The environment-droplet.yml file can be used to install the packages. Also, you need to setup some other packages which can be found in “/nas/vista-ssd01/batl/SFU/sfu-utils”
- Since we removed some of the images from dataset, we cannot extract h5 files and use them without manipulation. So, we need to extract and crop all images, then feed them to the network. You can use “driver.sh” which extracts the images. 
- After extracting the images and removing unwanted images from them, we group them into training, validation, and test set in a way that no same person is in two of these sets. The dataset file is Dataset.rar. The images in the dataroot should be in trainA, trainB, validA, validB, testA, and testB folders. Since we are not using the h5 files now the dataset mode is unaligned. 
- Now you can train the classifier and cyclegan using the scripts. The models store models every 5 epochs and also store the best model and latest model separately. We use resnet_6blocks+ for our generator which uses upsampling instead of the deconvolution. 
- To test the binary classifier, use “pad_binary_test.sh” The result is the temp.png file which contains the ROC curve. The output file shows the file indeces and whether they are TP, FP, TN, or FN. The output shows the positive class, which could be true positive or false positive. These samples are the ones which goes through the makeup removal. 
- To run the makeup removal part, you should put the images in the positive class in to the “testA” folder. After that you can run “pad_cycle_test.sh” to generate makeup removed images. The images are named “i-after.png” and “i-before.png”. “I” is the index.
- To run the matcher, you have to use matlab.
- Open “run_matcher.m” file. We use “matcher.m” function to match makeup and no-makeup images.

