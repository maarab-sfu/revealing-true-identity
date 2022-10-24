import time
from LightCNN.light_cnn import LightCNN_29Layers_v2
from PIL import Image
import copy 
from numpy import array
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import torchvision.transforms as transforms
from sklearn.metrics import roc_curve, auc  
import face_alignment
import imutils
from imutils import face_utils
from collections import OrderedDict
from batl_utils_modules.pad_algorithms.pad_algorithm import PADAlgorithm
from data import create_dataset
from data.base_dataset import BaseDataset, get_params, get_transform
from models import create_model
from util.visualizer import Visualizer
from options.train_options import TrainOptions
from options.valid_options import ValidOptions
from options.test_options import TestOptions
import torch
from util.codes import *
from util.util import tensor2im,plot_roc ,diagnose_network,enhance_img,get_stat,get_image_tensor,get_image
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import numpy as np
import os 
import random
import cv2
from skimage.measure import compare_ssim
from imutils.face_utils import FaceAligner
from imutils.face_utils import FACIAL_LANDMARKS_IDXS
from sklearn import metrics


class MakeupDetectionPADAlgorithm(PADAlgorithm):
    def __init__(self):
        self.description = "MakeupDetectionPADAlgorithm"
        self.name = "SFU_FACE_MAKEUP"         
        self.total_iters = 0                # the total number of training iterations
        self.val_total_iters = 0
    def get_name(self):
            return self.name
    def getMaskSSIM(self,before,after):    # SSIM based mask     
        (_, diff) = compare_ssim(before[0,:,:], after[0,:,:],full=True)
        diff = (diff * 255).astype("uint8")        
        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        k_size = 5
        kernel = np.ones((k_size,k_size),np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        # k_size = 5
        # kernel = np.ones((k_size,k_size),np.uint8)
        # thresh= cv2.dilate(thresh,kernel,iterations = 1)
        k_size = 7
        kernel = np.ones((k_size,k_size),np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        return thresh
    def getMaskDiff(self, before, after): # difference based mask
        clahe = cv2.createCLAHE(clipLimit=0.01,tileGridSize=(8,8))        
        after = after.astype(np.float)
        after = ((after-np.amin(after))*65536.0)/(np.amax(after)-np.amin(after))
        after = after.astype(np.uint16)
        after = clahe.apply(after[0,:,:])
        after = ((after-np.amin(after))*255.0)/(np.amax(after)-np.amin(after))
        after = after.astype(np.uint8)
        after = np.tile(after,(3,1,1))
        before = before.astype(np.int16)
        after = after.astype(np.int16)
        result = (before-after)
        result = ((result-np.amin(result))*255.0)/(np.amax(result)-np.amin(result))
        result = result.astype(np.uint8)
        result = clahe.apply(result[0,:,:])   
        mask = cv2.threshold(result,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
        temp_mask=np.copy(mask)
        k_size = 5
        kernel = np.ones((k_size,k_size),np.uint8)
        temp_mask = cv2.morphologyEx(temp_mask, cv2.MORPH_OPEN, kernel)
        k_size = 3
        kernel = np.ones((k_size,k_size),np.uint8)
        temp_mask = cv2.morphologyEx(temp_mask, cv2.MORPH_CLOSE, kernel)
        return temp_mask 
    def getMaskConst(self,image):   #constant mask based on facial landmarks
        image=np.moveaxis(image,0,-1)                
        shape     = self.predictor.get_landmarks(image)[0]
        overlay   = np.ones((image.shape[0],image.shape[1],3),dtype=np.uint8)*127
        minY=np.inf
        maxY=0
        midX=0
        minX=np.inf
        maxX=0
        for (i, name) in enumerate(FACIAL_LANDMARKS_IDXS.keys()):
            (j, k) = FACIAL_LANDMARKS_IDXS[name]
            pts = shape[j:k]
            if 'eyebrow' in name:  
                for pt in pts:
                    minY=min(minY,pt[1])
                continue
            if name == "jaw":
                for pt in pts:
                    minX=min(minX,pt[0])
                    maxX=max(maxX,pt[0])
                    if pt[1] > maxY:
                        maxY=pt[1]
                        midX=pt[0]
        (j, k) = FACIAL_LANDMARKS_IDXS['jaw']
        pts = shape[j:k]                
        pts = np.vstack((pts, (midX,max(3,minY-20))))
        pts = np.vstack((pts, (minX+8,max(3,minY-13))))
        pts = np.vstack((pts, (maxX-8,max(3,minY-13))))
        hull = cv2.convexHull(pts)    
        cv2.drawContours(overlay, [hull], -1, (255,255,255), -1)
        for (i, name) in enumerate(FACIAL_LANDMARKS_IDXS.keys()):    
            if name == 'jaw' or 'eyebrow' in name:
                continue        
            else:
                (j, k) = FACIAL_LANDMARKS_IDXS[name]
                pts = shape[j:k]
                hull = cv2.convexHull(pts)    
                cv2.drawContours(overlay, [hull], -1, (0,0,0), -1)
        kernel = np.ones((15,15),np.uint8)
        overlay[overlay==127]=0
        overlay = cv2.erode(overlay,kernel,iterations = 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))        
        overlay = cv2.morphologyEx(overlay, cv2.MORPH_OPEN, kernel, iterations=1)        
        return overlay[:,:,0]
    def summarize(self,phase,iters):
        if self.opt.dataset_mode == 'h5': 
            mask_temp = np.zeros((len(self.dataset.names),self.model.opt.crop_size,self.model.opt.crop_size),dtype=np.uint8)              
            
            mean=self.model.rgb_mean
            std= self.model.rgb_std            
            real_A_rgb=get_image(self.model.real_A,mean,std,grayscale=False)                                               
            fake_B_rgb=get_image(self.model.fake_B,mean,std,grayscale=False)      
            rec_A_rgb = get_image(self.model.rec_A,mean,std,grayscale=False)                
            self.writer.add_image(phase+'_source_rgb', real_A_rgb, iters)            
            self.writer.add_image(phase+'_reconstructed source_rgb_', rec_A_rgb, iters) 
            self.writer.add_image(phase+'_result_rgb_', fake_B_rgb, iters)                                         
            if self.opt.isTrain:
                real_B_rgb=get_image(self.model.real_B,mean,std,grayscale=False)
                rec_B_rgb =get_image(self.model.rec_B,mean,std,grayscale=False)
                fake_A_rgb=get_image(self.model.fake_A,mean,std,grayscale=False)
                if self.opt.netG=='drn':
                    warped_rgb=get_image(self.model.warped,mean,std,grayscale=False)                    
                    self.writer.add_image(phase+'_warped_rgb_', warped_rgb, iters)                                     
                self.writer.add_image(phase+'_reconstructed target_rgb_', rec_B_rgb, iters)
                self.writer.add_image(phase+'_target_rgb', real_B_rgb, iters)                
                self.writer.add_image(phase+'_backward_rgb_', fake_A_rgb, iters) 
            for i in range(len(self.dataset.names)):
                name=self.dataset.names[i]
                mean=get_stat(self.model.infra_mean,i,tile=True)                    
                std=get_stat(self.model.infra_std,i,tile=True)
                real_A=tensor2im(self.model.real_A[0][i+3].repeat((1,1,1,1)),mean=mean,std=std)                
                rec_A=tensor2im(self.model.rec_A[0][i+3].repeat((1,1,1,1)),mean=mean,std=std)                                
                fake_B=tensor2im(self.model.fake_B[0][i+3].repeat((1,1,1,1)),mean=mean,std=std)                                                                        
                self.writer.add_image(phase+'_source_'+name, real_A, iters)
                self.writer.add_image(phase+'_reconstructed source_'+name, rec_A, iters)                                        
                self.writer.add_image(phase+'_result_'+name, fake_B, iters)                                                
                if self.opt.isTrain:
                    real_B=tensor2im(self.model.real_B[0][i+3].repeat((1,1,1,1)),mean=mean,std=std)                    
                    rec_B=tensor2im(self.model.rec_B[0][i+3].repeat((1,1,1,1)),mean=mean,std=std)                    
                    fake_A=tensor2im(self.model.fake_A[0][i+3].repeat((1,1,1,1)),mean=mean,std=std)
                    if self.opt.netG == 'drn':
                        warped_infra=tensor2im(self.model.warped[0][i+3].repeat((1,1,1,1)),mean=mean,std=std)                        
                        self.writer.add_image(phase+'_warped_infra_'+name, warped_infra, iters)                                             
                    self.writer.add_image(phase+'_reconstructed target_'+name, rec_B, iters)
                    self.writer.add_image(phase+'_backward_'+name, fake_A, iters) 
                    self.writer.add_image(phase+'_target_'+name, real_B, iters)                                                                    
                    
                if self.opt.mask=='const':
                    mask=self.getMaskConst(real_A_rgb)
                elif self.opt.mask == 'diff':
                    mask=self.getMaskDiff(real_A,fake_B)
                    #mask=self.align(mask,real_A,real_A_rgb)
                elif self.opt.mask == 'ssim':
                    mask=self.getMaskSSIM(real_A,fake_B)
                elif self.opt.mask == 'union':
                    mask1=self.getMaskConst(real_A_rgb)
                    mask2=self.getMaskDiff(real_A,fake_B)
                    #mask2=self.align(mask2,real_A,real_A_rgb)
                    mask=np.clip(mask1+mask2, a_min = 0, a_max = 255) 
                mask_temp[i-1,:,:]=mask
                self.writer.add_image(phase+'_mask_'+name, np.tile(mask,(3,1,1)) , iters)
            #ignore other bands for now
            mask_temp[1,:,:]=mask_temp[0,:,:]
            mask_temp[2,:,:]=mask_temp[0,:,:]            
            source_rgb_unmasked=np.copy(real_A_rgb)
            real_A_rgb[mask_temp[0:3,:,:]==255]=255
            self.writer.add_image(phase+'_source_rgb_masked', real_A_rgb, iters)                        
            #self.writer.add_image(phase+'_average_mask', mask_temp, iters)                        
            self.writer.add_text(phase+'_source_id:',self.model.A_paths[0], iters)
            if self.opt.isTrain:
                self.writer.add_text(phase+'_target_id:',self.model.B_paths[0], iters)                                                                
            else:
                return (np.moveaxis(real_A_rgb,0,-1),np.moveaxis(mask_temp,0,-1),np.moveaxis(source_rgb_unmasked,0,-1),np.moveaxis(fake_B_rgb,0,-1))    
                            
        else:   #RGB
            rgb_mean=self.model.rgb_mean
            rgb_std= self.model.rgb_std
            real_A=get_image(self.model.real_A,rgb_mean,rgb_std,grayscale=False)
            rec_A=get_image(self.model.rec_A,rgb_mean,rgb_std,grayscale=False)
            fake_B=get_image(self.model.fake_B,rgb_mean,rgb_std,grayscale=False)                                                                                    
            self.writer.add_image(phase+'_source', real_A, iters)
            self.writer.add_image(phase+'_reconstructed source', rec_A, iters)                                    
            self.writer.add_image(phase+'_result', fake_B, iters)                                                
            if self.opt.isTrain:                
                rec_B=get_image(self.model.rec_B,rgb_mean,rgb_std,grayscale=False)
                fake_A=get_image(self.model.fake_A,rgb_mean,rgb_std,grayscale=False)
                real_B=get_image(self.model.real_B,rgb_mean,rgb_std,grayscale=False)  
                if self.opt.netG == 'drn':
                    warped_rgb=get_image(self.model.warped,rgb_mean,rgb_std,grayscale=False)                                     
                    self.writer.add_image(phase+'_warped_rgb_', warped_rgb, iters)                     
                self.writer.add_image(phase+'_reconstructed target', rec_B, iters)                        
                self.writer.add_image(phase+'_target', real_B, iters) 
                self.writer.add_image(phase+'_backward', fake_A, iters)  
            else:
                return( 0, 0, np.moveaxis(real_A,0,-1),np.moveaxis(fake_B,0,-1))

    def train(self, train_file_readers, save_path, valid_file_readers=None):    
            """
            Trains a model using the training and provided validation file readers and saves the trained model in the
            provided path file name.
            :param train_file_readers: list of file readers
            :param save_path: str, path and file name
            :param valid_file_readers: list of file readers
            :return: None
            """ 
            #self.predictor = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
            self.opt = TrainOptions().parse()   # get training options  
            os.makedirs(os.path.join(self.opt.checkpoints_dir,self.opt.name,'logs'), exist_ok=True)
            self.writer = SummaryWriter(os.path.join(self.opt.checkpoints_dir,self.opt.name,'logs'))      
            self.visualizer = Visualizer(self.opt)   # create a visualizer that display/save images and plots
            self.model = create_model(self.opt)      # create a model given opt.model and other options
            self.model.setup(self.opt)               # regular setup: load and print networks; create schedulers            
            self.opt.files = train_file_readers                         
            self.dataset = create_dataset(self.opt)  # create a dataset given opt.dataset_mode and other options            
            self.dataset_size = len(self.dataset)    # get the number of images in the dataset.
            print("dataset size: ", len(self.dataset))
            
            self.model.infra_mean=self.dataset.infra_mean
            self.model.infra_std=self.dataset.infra_std          
            self.model.rgb_mean=self.dataset.rgb_mean
            self.model.rgb_std =self.dataset.rgb_std                                         

            self.valid_opt = ValidOptions().parse()  # get test options
            if (self.opt.binary_test == 1):
                self.valid_opt = TestOptions().parse()  # get test options
            self.valid_opt.isTrain = True #evaluation is done when training
            self.valid_opt.files = valid_file_readers
            self.valid_opt.batch_size = 1    # test code only supports batch_size = 1
            self.valid_opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
            self.valid_opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
            self.valid_opt.display_id = -1 
            self.valid_dataset = create_dataset(self.valid_opt)
            if (self.opt.binary_test == 0):
                print('The number of training images = %d' % self.dataset_size)         

                print('########################################################################################')
                print('                                   TRAINING MODEL')
                print('########################################################################################')        
            best_loss = np.inf
            best_acc = 0.0
            best_val_auc = 0.0
            validation_tresh=0.0 #threshold to compute the accuracy of training set for the binary classifier, this threshold will be computed based on the validation set. In the first epoch we don't have a threshold because we didn't evaluate the model on the validation set yet. However, in the remaining epochs of training we use the best threshold from the validation of the previous  epoch.
            best_thresh=0.0
            for epoch in range(self.opt.epoch_count, self.opt.niter + self.opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
                epoch_start_time = time.time()  # timer for entire epoch                
                epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
                losses = OrderedDict()                
                self.model.set_epoch(epoch)
                if self.opt.model=='binary':
                    predictions_probs=[]
                    labels=[]
                for _, data in enumerate(self.dataset):  # inner loop within one epoch                                        
                    self.total_iters += self.opt.batch_size
                    epoch_iter += self.opt.batch_size
                    self.model.set_input(data)         # unpack data from dataset and apply preprocessing
                    self.model.optimize_parameters()   # calculate loss functions, get gradients, update network weights                    
                    losses[self.total_iters]  = self.model.get_current_losses()                       
                    if self.opt.model=='binary':
                        for t in self.model.target.detach().cpu().numpy().flatten(): 
                            labels.append(t)            
                        for t in self.model.score.detach().cpu().numpy().flatten():
                            predictions_probs.append(t)  
                                                   
                    if self.total_iters % self.opt.print_freq == 0 and (self.opt.model=='cycle_gan'):    # print training losses and save logging information to the disk                        
                        self.summarize('train', self.total_iters)
                            
                    if self.total_iters % self.opt.save_latest_freq == 0 and (self.opt.binary_test == 0):   # cache our latest model every <save_latest_freq> iterations
                        print('saving the latest model (epoch %d, total_iters %d)' % (epoch, self.total_iters))
                        save_suffix = 'iter_%d' % self.total_iters if self.opt.save_by_iter else 'latest'
                        self.model.save_networks(save_suffix, save_path) # will be overwritten
                    
                if epoch % self.opt.save_epoch_freq == 0 and (self.opt.binary_test == 0):              # cache our model every <save_epoch_freq> epochs
                    print('saving the model at the end of epoch %d, iters %d' % (epoch, self.total_iters))
                    self.model.save_networks('latest', save_path)
                    self.model.save_networks(epoch, save_path)                    
                res = self.evaluate(valid_file_readers)
                losses_val = res[1]
                
                for name in self.model.model_names:
                    print('gradient of {} is {}'.format(name,diagnose_network(eval('self.model.net'+name))))
                
                first=True
                summary=OrderedDict()
                if self.opt.model == 'cycle_gan':
                    summary['train_loss_G']=0
                for loss in losses.values():
                    if first:
                        first=False
                        for key, value in loss.items():
                            summary['train_loss_'+key]=0
                    for key, value in loss.items():
                        summary['train_loss_'+key]+=value
                        if (self.opt.model == 'cycle_gan') and 'D_' not in key:
                            summary['train_loss_G']+=value
                
                epoch_loss=0 
                if self.opt.model == 'cycle_gan':
                    epoch_loss=summary['train_loss_G']/len(losses)# for generator
                elif self.opt.model == 'binary':
                    epoch_loss=summary['train_loss_C']/len(losses)
                for key in list(summary.keys()):
                    summary[key]/=len(losses)
                    self.writer.add_scalar(key, summary[key], epoch)                    
                

                for key in list(losses_val.keys()): 
                    if 'total' not in key:
                        self.writer.add_scalar(key, losses_val[key], epoch)
                total_losses_val=losses_val['total']   
                improve=False             
                if total_losses_val < best_loss and (self.opt.binary_test == 0):
                    best_loss = total_losses_val
                    self.model.save_networks('best', save_path)  
                    improve=True
                        


                if self.opt.model == 'binary':
                    phase_ = "valid"
                    if (self.opt.binary_test == 1):
                        phase_ = "test"    
                    validation_tresh=res[2]
                    #validation_tresh = 0.6869
                    val_acc=res[0]
                    val_auc=res[3]
                    if val_auc>best_val_auc:
                        best_val_auc=val_auc
                    if val_acc > best_acc:
                        best_acc=val_acc #not necessarily at the best validation loss                    
                    if improve:
                        best_thresh=validation_tresh 
                    _,_,_,_,_,_,train_acc=self.get_accuracy(labels,predictions_probs,validation_tresh)
                    train_auc=metrics.roc_auc_score(labels, predictions_probs)
                    self.writer.add_scalar('training_auc', train_auc , self.model.epoch)
                    self.writer.add_scalar('training_acc', train_acc , self.model.epoch)
                    fpr, tpr, _ = metrics.roc_curve(labels,predictions_probs)
                    #ret=plot_roc(fpr,tpr,file_name='./checkpoints/'+self.opt.name+'/temp')
                    #self.writer.add_image('training_roc', np.moveaxis(ret[:,:,0:3],-1,0) , self.model.epoch)
                    if (self.opt.binary_test == 0):
                        print('Epoch %d \t Epoch Loss: %.4f Train AUC:%.4f Validation Loss:%.4f Best Validation Loss:%.4f Validation AUC:%.4f\
                            Best Validation AUC:%.4f Training Accuracy:%.4f Validation Accuracy:%.4f Best Validation Accuracy:%.4f Best Threshold:%.4f\
                            '%(epoch,epoch_loss,train_auc,total_losses_val,best_loss,val_auc,best_val_auc,train_acc,val_acc,best_acc,best_thresh))
                    else:
                        print('Epoch %d \t Epoch Loss: %.4f Train AUC:%.4f Test Loss:%.4f Best Test Loss:%.4f Test AUC:%.4f\
                         Best Test AUC:%.4f Training Accuracy:%.4f Test Accuracy:%.4f Best Test Accuracy:%.4f Best Threshold:%.4f\
                             '%(epoch,epoch_loss,train_auc,total_losses_val,best_loss,val_auc,best_val_auc,train_acc,val_acc,best_acc,best_thresh))
                elif self.opt.model=='cycle_gan':
                    print('Epoch %d \t Epoch Loss: %.4f Validation Loss:%.4f Best Loss:%.4f'%(epoch,epoch_loss,total_losses_val,best_loss))
                print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, self.opt.niter + self.opt.niter_decay, time.time() - epoch_start_time))
                self.model.update_learning_rate()                     # update learning rates at the end of every epoch.
       
    def evaluate(self, valid_file_readers):
        """
        This function calculates the validation loss based on the type of the model that is currently being trained.
        The validation loss is used to save the best model while training as this model will be used during testing(inference).
        The code of this function just replicates the same logic of the training optimization (e.g., check optimize_parameters in cycle_gan_model.py under models directory)
        """
        self.model.eval() #set to eval mode while validating                          
        losses = OrderedDict() 
        losses['total']=0
        if self.opt.model=='cycle_gan':            
            losses['val_loss_G']=0
            losses['val_loss_G_A']=0
            losses['val_loss_G_A_style']=0
            losses['val_loss_G_B']=0
            losses['val_loss_D_A']=0            
            losses['val_loss_D_B']=0            
            losses['val_loss_cycle_A']=0
            losses['val_loss_cycle_B']=0
            losses['val_loss_FM_A']=0
            losses['val_loss_FM_B']=0
            losses['val_loss_cycle_perceptual_A']=0
            losses['val_loss_cycle_perceptual_B']=0
            losses['val_loss_idt_A']=0
            losses['val_loss_idt_B']=0
            losses['val_loss_face_feature_A']=0
            losses['val_loss_face_feature_B']=0
            losses['val_loss_D_makeup']=0
            losses['val_loss_sparse']=0
        elif self.opt.model=='binary':
            losses['val_loss_C']=0
            labels=[]
            predictions_probs=[]

            # losses['val_loss_idt_A']=0
            # losses['val_loss_idt_B']=0
            # losses['val_loss_face_feature_A']=0
            # losses['val_loss_face_feature_B']=0
            

        with torch.no_grad():
            for iteration, data in enumerate(self.valid_dataset):
                self.val_total_iters+=1
                self.model.set_input(data)  # unpack data from data loader
                self.model.test()           # run inference
                if self.opt.model == 'binary':       
                    for t in self.model.target.detach().cpu().numpy().flatten(): 
                        labels.append(t)            
                    for t in self.model.score.detach().cpu().numpy().flatten():
                        predictions_probs.append(t)            
                    val_loss_C=self.model.criterionClassifier(self.model.score,self.model.target) 
                    losses['val_loss_C']+=val_loss_C
                    losses['total']+=val_loss_C
                elif self.opt.model=='cycle_gan':
                    if iteration%max(1,len(self.valid_dataset)//10)==0:
                        self.summarize('val', self.val_total_iters)
                    # GAN loss D_A(G_A(A))
                    out_D_A=self.model.netD_A(self.model.fake_B)
                    val_loss_G_A=self.model.criterionGAN(out_D_A, True).item()
                    val_loss_G_A_style = 0                    
                    if self.opt.netG == 'drn':
                        out_D_makeup_A = self.model.netD_makeup(self.model.real_A, self.model.fake_A)                    
                        val_loss_G_A_style = self.model.criterionGAN(out_D_makeup_A, True).item()
                    out_D_B=self.model.netD_B(self.model.fake_A)
                    val_loss_G_B=self.model.criterionGAN(out_D_B, True).item()
                    
                    val_loss_cycle_A=self.model.criterionCycle(self.model.rec_A, self.model.real_A).item() * self.model.opt.lambda_A
                    val_loss_cycle_B=self.model.criterionCycle(self.model.rec_B, self.model.real_B).item() * self.model.opt.lambda_B

                    val_loss_sparse=self.model.criterionCycle(self.model.fake_A, self.model.real_B).item() * 0.0
                    
                    val_loss_cycle_perceptual_A=(1-self.model.criterionPerceptualCycle(self.model.rec_A, self.model.real_A).item()) * self.model.opt.lambda_A * self.model.opt.perceptual_factor
                    val_loss_cycle_perceptual_B=(1-self.model.criterionPerceptualCycle(self.model.rec_B, self.model.real_B).item()) * self.model.opt.lambda_B * self.model.opt.perceptual_factor                    
                    
                    if self.model.opt.lambda_identity > 0:
                        val_loss_idt_A=self.model.criterionIdt(self.model.idt_A, self.model.real_B).item() * self.model.opt.lambda_B * self.model.opt.lambda_identity
                        val_loss_idt_B=self.model.criterionIdt(self.model.idt_B, self.model.real_A).item() * self.model.opt.lambda_A * self.model.opt.lambda_identity      
                    else:
                        val_loss_idt_A=0
                        val_loss_idt_B=0
                    losses['val_loss_G_A'] += val_loss_G_A
                    # GAN loss D_B(G_B(B))
                    losses['val_loss_G_A_style'] += val_loss_G_A_style
                    # GAN loss D_B(G_B(B))
                    losses['val_loss_G_B'] += val_loss_G_B
                    # Forward cycle loss || G_B(G_A(A)) - A||
                    losses['val_loss_cycle_A'] += val_loss_cycle_A
                    # Backward cycle loss || G_A(G_B(B)) - B||
                    losses['val_loss_cycle_B'] += val_loss_cycle_B
                    
                    losses['val_loss_sparse'] += val_loss_sparse

                    # Forward cycle loss || G_B(G_A(A)) - A||
                    losses['val_loss_cycle_perceptual_A'] += val_loss_cycle_perceptual_A
                    # Backward cycle loss || G_A(G_B(B)) - B||
                    losses['val_loss_cycle_perceptual_B'] += val_loss_cycle_perceptual_B
                    # combined loss and calculate gradients
                    losses['val_loss_idt_A'] += val_loss_idt_A
                    # G_B should be identity if real_A is fed: ||G_B(A) - A||                
                    losses['val_loss_idt_B'] += val_loss_idt_B

                    
                    val_loss_FM_A = 0
                    val_loss_FM_B = 0
                    
                    out_D_A_real = self.model.netD_A(self.model.real_B)
                    out_D_B_real = self.model.netD_B(self.model.real_A)
                    if self.opt.netD == 'dilated':                               
                        for i in range(0,len(out_D_A_real[1])):
                            val_loss_FM_A += self.model.criterionFeatureMatch(out_D_A[1][i], out_D_A_real[1][i].detach()).item()
                            val_loss_FM_B += self.model.criterionFeatureMatch(out_D_B[1][i], out_D_B_real[1][i].detach()).item()
                        val_loss_FM_A /= len(out_D_A_real[1])
                        val_loss_FM_B /= len(out_D_B_real[1])
                        val_loss_FM_A *= self.model.opt.lambda_feature
                        val_loss_FM_B *= self.model.opt.lambda_feature
                    losses['val_loss_FM_A'] += val_loss_FM_A
                    losses['val_loss_FM_B'] += val_loss_FM_B

                    val_loss_face_feature_A = 0
                    val_loss_face_feature_B = 0                       
                    if self.opt.lambda_face_identity > 0.0:                      
                        mean=self.model.rgb_mean
                        std= self.model.rgb_std                        
                        tmp_rec_A =get_image_tensor(self.model.rec_A,mean,std)
                        tmp_real_A=get_image_tensor(self.model.real_A,mean,std)
                        tmp_rec_B =get_image_tensor(self.model.rec_B,mean,std)
                        tmp_real_B=get_image_tensor(self.model.real_B,mean,std)
                        val_loss_face_feature_A = self.model.criterionFaceFeature(self.model.face_model(tmp_rec_A[:,:,::2,::2])[1], self.model.face_model(tmp_real_A[:,:,::2,::2])[1]).item() * self.opt.lambda_face_identity                
                        val_loss_face_feature_B = self.model.criterionFaceFeature(self.model.face_model(tmp_rec_B[:,:,::2,::2])[1], self.model.face_model(tmp_real_B[:,:,::2,::2])[1]).item() * self.opt.lambda_face_identity
                        losses['val_loss_face_feature_A']+=val_loss_face_feature_A
                        losses['val_loss_face_feature_B']+=val_loss_face_feature_B
                    losses['val_loss_G'] +=     val_loss_G_A+val_loss_G_A_style+val_loss_G_B+val_loss_cycle_A+val_loss_cycle_B+val_loss_idt_A+val_loss_idt_B + val_loss_cycle_perceptual_A + val_loss_cycle_perceptual_B+val_loss_FM_A+val_loss_FM_B+val_loss_face_feature_A + val_loss_face_feature_B+ val_loss_sparse
                    # we need generators with minimum loss
                    losses['total']=losses['val_loss_G']

                    lbl_true = True                
                    pred_fake =  self.model.netD_A(self.model.fake_B_pool.query(self.model.fake_B))
                    val_loss_D_A_Fake=self.model.criterionGAN(pred_fake,not(lbl_true)).item()
                    val_loss_D_A_Real=self.model.criterionGAN(out_D_A_real, lbl_true).item()                
                    losses['val_loss_D_A'] += ((val_loss_D_A_Real+val_loss_D_A_Fake)*0.5)

                    # GAN loss D_B(G_B(B))
                    pred_fake =  self.model.netD_B(self.model.fake_A_pool.query(self.model.fake_A))
                    val_loss_D_B_Fake=self.model.criterionGAN(pred_fake,not(lbl_true)).item()
                    val_loss_D_B_Real=self.model.criterionGAN(out_D_B_real, lbl_true).item()                
                    losses['val_loss_D_B'] += ((val_loss_D_B_Real+val_loss_D_B_Fake)*0.5) 

                    val_loss_D_makeup_fake = 0
                    val_loss_D_makeup_real = 0
                    if self.opt.netG == 'drn':
                        pred_makeup_real= self.model.netD_makeup(self.model.real_A, self.model.warped)
                        val_loss_D_makeup_fake = self.model.criterionGAN(out_D_makeup_A, not(lbl_true)).item()
                        val_loss_D_makeup_real = self.model.criterionGAN(pred_makeup_real, lbl_true).item()
                    losses['val_loss_D_makeup'] += ((val_loss_D_makeup_real+val_loss_D_makeup_fake)*0.5) 
        for key in list(losses.keys()):
            losses[key]/=len(self.valid_dataset)
        self.model.train() #return to train mode before going back to training loop
        if self.opt.model == 'binary':
            phase_ = "valid"
            if (self.opt.binary_test == 1):
                phase_ = "test"
            print(phase_+" set len :{}".format(len(self.valid_dataset)))
            #https://stackoverflow.com/questions/28719067/roc-curve-and-cut-off-point-python
            fpr, tpr, thresholds = metrics.roc_curve(labels,predictions_probs)
            print("thresholds: ",thresholds)
            print("fpr: ",fpr)
            print("tpr: ",tpr)
            auc=metrics.roc_auc_score(labels, predictions_probs)             
            eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
            if (self.opt.binary_test == 0):
                optimal_threshold = interp1d(fpr, thresholds)(eer)
            else:
                optimal_threshold = self.opt.binary_thresh
            print("optimal_threshold: ", optimal_threshold)
            predictions_probs=array(predictions_probs)
            labels=array(labels)
            pred_bona_max=predictions_probs[labels==BF][np.argmax(predictions_probs[labels==BF])]
            pred_attack_min=predictions_probs[labels==ATTACK][np.argmin(predictions_probs[labels==ATTACK])]            
            print(optimal_threshold)
            positive,negative,true_positive,true_negative,false_positive,false_negative,acc=self.get_accuracy(labels,predictions_probs,optimal_threshold, phase_)
            self.writer.add_scalar(phase_+"_auc", auc , self.model.epoch)
            self.writer.add_scalar(phase_+"_accuracy", acc , self.model.epoch)
            ret=plot_roc(fpr,tpr,file_name='./checkpoints/'+self.opt.name+'/temp')  
            self.writer.add_image(phase_+"_roc", np.moveaxis(ret[:,:,0:3],-1,0) , self.model.epoch)
            print("pos:{},neg:{},true_pos:{},true_neg:{},false_pos:{},false_neg:{}".format(positive,negative,true_positive,true_negative,false_positive,false_negative))
            print("Results on "+phase_+" set: TPR:{}, FPR:{}, TNR:{}, FNR:{}, Accuracy:{}".format(true_positive/positive,false_positive/negative,true_negative/negative,false_negative/positive,acc))
            print("max prob for bona:{} min prob for attack:{}".format(pred_bona_max,pred_attack_min))
            return acc,losses,optimal_threshold,auc
        else:
            return 0,losses
    def get_accuracy(self, labels, predictions_probs, optimal_threshold, phase = 'train'):
        """
        Calculate the accruacy of prediction while validating or testing the binary classifier and the whole system (i.e., classifier + makeup removal + matcher)
        """
        true_positive  = 0.0 #attack is predicted as attack
        true_negative  = 0.0 #bona fide is predicted as bona fide
        false_positive = 0.0 #bona fide is predicted as attack
        false_negative = 0.0 #attack is predicted as bona fide
        positive = 0.0
        negative = 0.0
        for i,label in enumerate(labels):
            score_orig=predictions_probs[i]
            if(score_orig>=optimal_threshold):
                score=ATTACK
            else:
                score=BF            
            if label == ATTACK:
                positive += 1
                if score == ATTACK:                    
                    true_positive += 1
                    if phase == 'test':
                        print("true positive: ", self.model.paths[i])
                else:                    
                    false_negative += 1
                    #if phase == 'test':
                        #print("false negative: ", self.model.paths[i])
            else:
                negative += 1            
                if score == BF:
                    true_negative +=1
                    #if phase == 'test':
                        #print("true negative: ", self.model.paths[i])
                else:
                    false_positive +=1
                    if phase == 'test':
                        print("false positive: ", self.model.paths[i])                                 
        return positive,negative,true_positive,true_negative,false_positive,false_negative,(true_positive+true_negative)/(positive+negative)

    def load_model(self, load_path):
            """
            Loads an algorithm including the trained model from a previous training session.
            :param load_path: str, path and file name to the saved trained algorithm+model
            :return: None
            """
            self.opt = TestOptions().parse()  # get test options assuming we are going to call predict_pa since retrain is not needed now
            self.opt.num_threads = 0   # test code only supports num_threads = 1
            self.opt.batch_size = 1    # test code only supports batch_size = 1
            self.opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
            self.opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
            self.opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.            
            self.model = create_model(self.opt)      # create a model given opt.model and other options            
            self.model.setup(self.opt, load_path)                
    
    def predict_pa(self, file):
        """
        Predict whether a sample is an attack or not.
        If the binary classifier only is being tested, the attack(positive) class contains makeup attacks and normal makeup while the negative class contains no makeup samples.
        If the whole system is being tested, the attack (positive) class contains makeup attacks only and the negative class contains normal makeup and no makeup samples.
        This function returns a prediction score and a threshold such that if the prediction score is larger than or equal the threshold, then the sample is considered an attack.        
        """
        self.opt = TestOptions().parse()  # get test options
        # in case predict_pa is called directly without loading model
        self.opt.num_threads = 0   # test code only supports num_threads = 1
        self.opt.batch_size = 1    # test code only supports batch_size = 1
        self.opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
        self.opt.no_flip = True    # no flip; comment this line cp(u results on flipped images are needed.
        self.opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
        self.opt.files = [file]        
        self.dataset = create_dataset(self.opt)  # create a dataset given opt.dataset_mode and other options
        self.dataset_size = len(self.dataset)    # get the number of images in the dataset.
        opt_binary = copy.deepcopy(self.opt) # The same testing options are used for the binary classifier except a few options such as the crop/load sizes.
        #The binary classifier was trained on 32x32 downsampled face images since the dataset size is small.
        opt_binary.crop_size=32
        opt_binary.load_size=32
        opt_binary.model='binary'                
        if self.opt.model=='cycle_gan':
            self.binary_dataset = create_dataset(opt_binary)            

        if not hasattr(self, 'model'):
            self.model = create_model(self.opt)      # create a model given opt.model and other options
            self.model.setup(self.opt)               # regular setup: load and print networks; create schedulers
            
            
        self.model.infra_mean=self.dataset.infra_mean
        self.model.infra_std=self.dataset.infra_std        
        self.model.rgb_mean=self.dataset.rgb_mean
        self.model.rgb_std =self.dataset.rgb_std   
        flag = 0
        if not hasattr(self, 'binary_model') and self.opt.model=='cycle_gan': #initialize and load the binary model in case we are testing the whole system
            flag = 1
        if self.opt.model=='cycle_gan' and self.opt.matcher == 'CNN' and not hasattr(self,'face_model'):
            self.face_model = torch.nn.DataParallel(LightCNN_29Layers_v2(num_classes=80013)).cuda()                               
            self.face_model.eval()
            self.face_model.load_state_dict((torch.load('/nas/home/aminarab/batl-utils/LightCNN/LightCNN_29Layers_V2_checkpoint.pth'))['state_dict'])

        # test with eval mode. This only affects layers like batchnorm and dropout.
        self.model.eval()
        os.makedirs(os.path.join(self.opt.checkpoints_dir,self.opt.name,'test-logs'), exist_ok=True)
        self.writer = SummaryWriter(os.path.join(self.opt.checkpoints_dir,self.opt.name,'test-logs'))      
        score = 0
                           
        binary_thresh = self.opt.binary_thresh 
        if self.opt.model=='cycle_gan':
            thresh = self.opt.matcher_thresh #Neurotech threshold will be determined from the validation set 
        for i, data in enumerate(self.dataset): #it's one file
            self.total_iters += self.opt.batch_size             
            if self.opt.model=='cycle_gan' and flag == 0:
                #We first need to know if the sample has any makeup or not. 
                # This is decided by the binary classifier which should predict whether the sample has makeup or not. 
                # If the sample is predicted to not having makeup by the classifier, then it bypasses the makeup removal. 
                # If the sample has makeup as decided by the classifier, then it goes through makeup removal and the matcher should decide whether the difference between the before and after makeup removal is significant or not.
               for _, binary_data in enumerate(self.binary_dataset): #it's one file                
                    self.binary_model.set_input(binary_data)
                    self.binary_model.test()
                    score = self.binary_model.score.detach().cpu().item()                
               prediction = BF      

               if score>=binary_thresh: #This means that the classifier decided that the sample has makeup and should go through the makeup removal network.
                   prediction=ATTACK
               if prediction==BF: # do not go through makeup removal pipeline if the binary classifier has predicted that this sample does not have makeup at all
                    score  = BF
                    thresh = binary_thresh
               else:                    
                    self.model.set_input(data)  # unpack data from data loader
                    self.model.test()           # run inference
                    source_rgb,mask_temp,source_rgb_unmasked,result_rgb=self.summarize('test', self.total_iters)                
                    cv2.imwrite('source_rgb.png', cv2.cvtColor(source_rgb, cv2.COLOR_RGB2BGR))
                    #cv2.imwrite('mask.png',mask_temp)
                    #cv2.imwrite('source_rgb_unmasked.png',cv2.cvtColor(source_rgb_unmasked, cv2.COLOR_RGB2BGR))
                    cv2.imwrite('result_rgb.png',cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR))
                    if self.opt.method=='inpaint': # This is old testing code when we were inpainting the makeup regions of the RGB image based on a mask computed from the infra images
                        os.system("cp source_rgb.png $HOME/edge-connect")
                        os.system("cp mask.png $HOME/edge-connect")
                        os.system("cd $HOME/edge-connect && python test.py --input source_rgb.png --mask mask.png --output . --checkpoints checkpoints/celeba")               
                        os.system("cp $HOME/edge-connect/source_rgb.png ./inpainted.png")
                        inpainted=cv2.imread('inpainted.png')
                        inpainted=cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
                        self.writer.add_image('test_inpainted', np.moveaxis(inpainted,-1,0) , self.total_iters)               
                        os.system("cp inpainted.png /nas/vista-ssd01/batl/SFU/Matcher/src/after.png") #copy the image after inpainting the RGB image to the directory where the matcher code exists.               
                    else:
                        os.system("cp result_rgb.png /nas/vista-ssd01/batl/SFU/Matcher/src/after.png") # copy the image after makeup removal to the directory where the matcher code exists.               
                    score=ATTACK # assume it's attack if face matcher did not detect face                    
                    if self.opt.matcher == 'CNN':
                        source_rgb_unmasked = Image.fromarray(source_rgb_unmasked)
                        result_rgb          = Image.fromarray(result_rgb)
                        transform_params    = get_params(self.opt, source_rgb_unmasked.size)                                        
                        transform           = get_transform(self.opt, transform_params, custom_mean=self.model.rgb_mean, custom_std=self.model.rgb_std)        
                        before_features     = self.face_model(transform(source_rgb_unmasked))[1]
                        after_features      = self.face_model(transform(result_rgb))[1]
                        score               = 1.0 - torch.nn.functional.cosine_similarity(before_features,after_features) 

                    else:
                        os.system("cp source_rgb_unmasked.png /nas/vista-ssd01/batl/SFU/Matcher/src/before.png") # copy the source RGB image before makeup removal to the directory where the matcher code exists.
                        os.system('cd /nas/vista-ssd01/batl/SFU/Matcher/src && java -cp .:/nas/vista-ssd01/batl/SFU/Matcher/\\* VerifyFace "before.png" "after.png" >> score.txt') # Run the matcher code and take its output in a text file called score.txt
                        os.system('cp /nas/vista-ssd01/batl/SFU/Matcher/src/score.txt .') #copy score.txt to the current directory
                        #open and read the score from score.txt
                        f_read = open("score.txt", "r")
                        last_line = f_read.readlines()[-1]
                        f_read.close()
                        try:
                            #The matcher score can go from 0 (no match) to infinity (perfect match)
                            #Based on the documentation of the matcher a score of 100 corresponds to false acceptance rate less than 0.000001% so we used it as our maximum matcher score.
                            #Since the calling function expects a score between 0 (bona fide) and 1 (attack) we perform the below transformation on the matcher's score.
                            score=1.0-min(100.0,int(last_line.split(",")[0].split(" ")[2]))/100.0  
                        except:
                            #In case the matcher did not produce a score which is due to errors in detecting the face (e.g., the resulting image after makeup removal was very distorted) we consider it as attack.
                            print('face not found in inpainted image for :{}'.format(self.model.A_paths[0]))
                    self.writer.add_scalar('score', score , self.total_iters)  
            elif self.opt.model=='cycle_gan' and flag == 1:
                # prediction = file.ground_truth_classifier
                self.model.set_input(data)  # unpack data from data loader
                self.model.test()           # run inference
                source_rgb,mask_temp,source_rgb_unmasked,result_rgb=self.summarize('test', self.total_iters)             
                after_name = str(i)+"-after"+".png"
                before_name = str(i)+"-before"+".png"
                cv2.imwrite(before_name, cv2.cvtColor(source_rgb_unmasked, cv2.COLOR_RGB2BGR))
                #cv2.imwrite('mask.png',mask_temp)
                #cv2.imwrite('source_rgb_unmasked.png',cv2.cvtColor(source_rgb_unmasked, cv2.COLOR_RGB2BGR))
                cv2.imwrite(after_name,cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR))
                #os.system("cp result_rgb.png /nas/vista-ssd01/batl/SFU/Matcher/src/"+after_name) # copy the image after makeup removal to the directory where the matcher code exists.
                score=ATTACK # assume it's attack if face matcher did not detect face
                    
            elif self.opt.model=='binary':
                self.model.set_input(data)  # unpack data from data loader
                self.model.test()           # run inference
                score = self.model.score
                score = score.detach().cpu().item()                          
                thresh = binary_thresh
            
        return (score,thresh)            
            



    def __validate_data__(self, file):
        return
        
if __name__ == "__main__":
    print("hi")

               
