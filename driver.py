import sys
import os
from scipy.optimize import brentq
from face_data_extractor.face_data_extractor import *
from scipy.interpolate import interp1d
from custom_aligner.align_mouth import FaceAligner
from numpy import array
import face_alignment
import tables
import argparse
from imutils.face_utils import rect_to_bb
import argparse
import imutils
from PIL import Image
import random
import imageio
import numpy as np
import scipy.misc
import cv2
import traceback
from data import create_dataset
from options.test_options import TestOptions
from options.valid_options import ValidOptions
import logging
from batl_utils_modules.dataIO.odin_h5_dataset import OdinH5Dataset, Odin5MultiDataset
from batl_utils_modules.pad_algorithms.pad_algorithm import *
import warnings
from util.codes import *
from MakeupPAD import MakeupDetectionPADAlgorithm
import torch.multiprocessing
from util.util import enhance_img,plot_roc
import csv
from collections import OrderedDict
from sklearn import metrics

args=[]

if __name__ == "__main__":    
    torch.multiprocessing.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser(description='Arguments to retrieve infra/RGB data.')
    parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--prob', type=float, default=0.05, help='label switching probability for discriminator')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam') #https://medium.freecodecamp.org/how-to-pick-the-best-learning-rate-for-your-machine-learning-project-9c28865039a8
    parser.add_argument('--print_freq', type=int, default=25, help='frequency of showing training results on console')
    parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
    parser.add_argument('--data',default="ST3",help='ST4 or ST3 or combined')
    parser.add_argument('--infra_bands', type=str, default='swir-1050nm#swir-1200nm#swir-1450nm',help='list of infra bands to be used')
    parser.add_argument('--dataroot', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
    parser.add_argument('--dataset_mode', type=str, default='h5', help='chooses how datasets are loaded. [unaligned | aligned | single | colorization | h5]')
    parser.add_argument('--mode',default="train",help='train or test')    
    parser.add_argument('--mask', type=str, default='diff',help='diff or ssim or const')
    parser.add_argument('--load_size', type=int, default=143, help='scale images to this size')
    parser.add_argument('--crop_size', type=int, default=128, help='then crop to this size')
    parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
    parser.add_argument('--input_nc', type=int, default=6, help='# of input image channels: 3 for RGB and 1 for infra') 
    parser.add_argument('--output_nc', type=int, default=6, help='# of output image channels: 3 for RGB or infra')
    parser.add_argument('--model', type=str, default='binary', help='chooses which model to use. [cycle_gan | pix2pix | test | colorization| binary]')       
    parser.add_argument('--netG', type=str, default='resnet_9blocks', help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
    parser.add_argument('--netD', type=str, default='basic', help='specify discriminator architecture [basic | n_layers | pixel| dilated]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
    parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--stddev', type=float, default=0.1, help='standard deviation of gaussian noise which is added to the discriminator''s input')
    parser.add_argument('--thresh_calc', action='store_true', help='use eval mode during test time.')
    parser.add_argument('--include_normal_makeup', action='store_true', help='include bona fide normal makeup in the domain of makeup while training CycleGAN')    
    parser.add_argument('--blur', action='store_true', help='blur infra bands')  
    parser.add_argument('--attack_only', action='store_true', help='attack vs no makeup only')        
    parser.add_argument('--matcher_thresh', type=float, default=0.5, help='matcher threshold')
    parser.add_argument('--binary_thresh', type=float, default=0.5, help='matcher threshold')
    parser.add_argument('--perceptual_factor', type=float, default=0.0, help='factor of perceptual loss as compared to L1 loss')
    parser.add_argument('--lambda_identity', type=float, default=0.0, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
    parser.add_argument('--lambda_class', type=float, default=0.0, help='classification loss weight')
    parser.add_argument('--lambda_feature', type=float, default=2.0, help='feature matching weight')
    parser.add_argument('--lambda_face_identity', type=float, default=0.03, help='face identity weight')
    parser.add_argument('--niter', type=int, default=200, help='# of iter at starting learning rate')
    parser.add_argument('--niter_decay', type=int, default=800, help='# of iter to linearly decay learning rate to zero')
    parser.add_argument('--method', type=str, default='inpaint', help='which method to use for detecting attacks [inpaint | rgb]. inpaint means we use the mask computed from difference between infra before and after makeup removal and applied to rgb to be inpainted. rgb means we use the rgb image after makeup removal from CycleGAN and pass it directly to the matcher to be compared with the rgb image before makeup removal.')
    parser.add_argument('--binary_classifier_dir', type=str, default='/ISI_binary/', help='models are saved here')
    parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
    parser.add_argument('--matcher', type=str, default='CNN', help='which matcher to use [CNN | VeriLook]')
    parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
    parser.add_argument('--test_fold', type=int, default=5, help='choose the test fold in 5 fold cross-validation')
    parser.add_argument('--binary_test', type=int, default=0, help='Are we testing the binary classifier')
    args = parser.parse_args()
    alg=MakeupDetectionPADAlgorithm()
    out_path='./checkpoints/'+args.name
    if args.mode=='train':                    
        alg.train([],out_path,[]) # save best and latest models in out_path
    elif args.mode=='test':
        alg.load_model(out_path) #loads best model from out_path
        test_dataset = create_dataset(ValidOptions().parse())
        dataset_size = len(test_dataset)
        if args.model == 'binary':
            true_positive  = 0.0 #attack is predicted as attack
            true_negative  = 0.0 #bona fide is predicted as bona fide
            false_positive = 0.0 #bona fide is predicted as attack
            false_negative = 0.0 #attack is predicted as bona fide
            positive = 0.0
            negative = 0.0
            # random.seed("USC/ISI-BATL")                
            # random.shuffle(test_dataset) 
            labels=[]
            prediction_probs=[]
            for i, sample_reader in enumerate(test_dataset):
                # gt = sample_reader.ground_truth_classifier  
                gt = sample_reader['label'].numpy() 
                (score_orig,threshold) = alg.predict_pa(sample_reader)  
                print(score_orig,' ', threshold)
                            
                labels.append(gt)             
                prediction_probs.append(score_orig)
                score=BF
                if(score_orig>=args.binary_thresh):
                    score=ATTACK     
                print('score: ', score, 'gt: ', sample_reader['label'].numpy())                                                                 
                if gt == ATTACK:
                    positive += 1
                    if score == ATTACK:
                        print('attack sample identified as attack, score:{}'.format(score_orig))
                        true_positive += 1
                    else:
                        print('attack sample as bona fide, score:{}'.format(score_orig))
                        false_negative += 1
                else:
                    negative += 1            
                    if score == BF:
                        print('bona fide identified as bona fide, score:{}'.format(score_orig))
                        true_negative +=1
                    else:
                        false_positive +=1 
                        print('bona fide identified as attack, score:{}'.format(score_orig))
                sys.stdout.flush()
            print("Results on test set: TPR:{}, FPR:{}, TNR:{}, FNR:{}, Accuracy:{}".format(true_positive/positive,false_positive/negative,true_negative/negative,false_negative/positive,(true_negative+true_positive)/(positive+negative)))
            fpr, tpr, thresholds = metrics.roc_curve(labels,prediction_probs)            
            plot_roc(fpr,tpr,file_name='./checkpoints/'+args.name+'/ROC_'+args.model+'_'+str(args.binary_thresh)+'_'+str(args.matcher_thresh))            
            eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
            optimal_threshold = interp1d(fpr, thresholds)(eer)
            prediction_probs=array(prediction_probs)
            labels=array(labels)
            pred_bona_max=prediction_probs[labels==BF][np.argmax(prediction_probs[labels==BF])]
            pred_attack_min=prediction_probs[labels==ATTACK][np.argmin(prediction_probs[labels==ATTACK])]
            print("{}--max prob for bona:{} min prob for attack:{}".format(args.model,pred_bona_max,pred_attack_min))
            print("optimal threshold for {} :{}".format(args.model,optimal_threshold))
        else:
            (score_orig,threshold) = alg.predict_pa(test_dataset)
        

