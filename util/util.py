"""This module contains simple helper functions """
from __future__ import print_function
import matplotlib.pyplot as plt
from sklearn import metrics
import torch
import numpy as np
from PIL import Image
import os
import torchvision.transforms as transforms
import cv2
import face_alignment
from custom_aligner.align_mouth import FaceAligner
import collections

def face_part(input_img, preds, index):
    img = np.zeros((400, 400, 3))
    crop = 140/2
    img[:,:,:] = input_img
    lips_xy = np.mean(preds[48:67],axis=0).astype(np.int)
    r_eye_xy = np.mean(np.concatenate((preds[17:21],preds[36:41]), axis = 0),axis=0).astype(np.int)
    l_eye_xy = np.mean(np.concatenate((preds[22:26],preds[42:47]), axis = 0),axis=0).astype(np.int)

    lips = img[lips_xy[1]-crop:lips_xy[1]+crop, lips_xy[0]-crop:lips_xy[0]+crop, :]
    r_eye = img[r_eye_xy[1]-crop:r_eye_xy[1]+crop, r_eye_xy[0]-crop:r_eye_xy[0]+crop, :] 
    l_eye = img[l_eye_xy[1]-crop:l_eye_xy[1]+crop, l_eye_xy[0]-crop:l_eye_xy[0]+crop, :]
    
    face = np.copy(img)
    face[lips_xy[1]-crop:lips_xy[1]+crop, lips_xy[0]-crop:lips_xy[0]+crop, :] = 0
    face[r_eye_xy[1]-crop:r_eye_xy[1]+crop, r_eye_xy[0]-crop:r_eye_xy[0]+crop, :] = 0
    face[l_eye_xy[1]-crop:l_eye_xy[1]+crop, l_eye_xy[0]-crop:l_eye_xy[0]+crop, :] = 0

    return lips,r_eye,l_eye,face

def get_image(input,mean,std,grayscale=True):
    """
    Returns the RGB component from a tensor as a numpy array.
    This is used with tensorboard to save images.
    """
    tmp=tensor2im(input[:,0:3,:,:],mean=mean,std=std)
    if grayscale:
        tmp=np.moveaxis(tmp,0,-1)
        tmp=np.reshape(cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY),(tmp.shape[0],tmp.shape[1],1))
    return tmp
def get_image_tensor(input_image,mean,std):
    """
    Converts an RGB tensor to a grayscale tensor to be used with LightCNN network which expects a grayscale face image from which it extracts facial features.
    The facial features are used to compute the face identity loss to ensure that the person's identity is not changed.
    """
    if mean is None:
        mean=0.5
    else:
        mean = torch.as_tensor(mean).float().cuda()
    if std is None:
        std=0.5    
    else:    
        std  = torch.as_tensor(std).float().cuda()
    input_image = (input_image[:,0:3,:,:]+ (mean/std)) * std  * 255.0  
    input_image = torch.clamp(input_image, min = 0, max = 255) 
    input_image = 0.2989 * input_image[:,0,:,:] + 0.5870 * input_image[:,1,:,:] + 0.1140 * input_image[:,2,:,:]
    input_image = input_image/255.0
    input_image = input_image.unsqueeze(0)    
    return input_image
def performance_metrics(fpr,tpr, percentage):
    for i, rate in enumerate(fpr):
        if rate>percentage:
            break;
    return tpr[i-1], fpr[i-1]
def plot_roc(fpr,tpr,file_name=''):
    """
    Saves the ROC curve with different formats (png, pdf, and eps) given false positve rate and true positive rate arrays that were calculated 
    beforehand based on the prediction labels and prediction probabilities.
    """
    roc_auc = metrics.auc(fpr, tpr)
    [TheMetrics, Att] = performance_metrics(fpr,tpr,0.002)    
    # print('fpr:{}'.format(fpr))
    # print('tpr:{}'.format(tpr))
    # print('roc:{}'.format(roc_auc))
    # plt.rcParams.update({'lines.linewidth': 6})
    plt.rcParams.update({'font.size': 16})
    plt.gcf().subplots_adjust(bottom=0.15, left=0.25)
    plt.title('Receiver Operating Characteristic')
    #plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f, TPR@%0.2f%%FPR = %0.2f' % (roc_auc, Att, TheMetrics))
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % (roc_auc))
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    # plt.xlim([0, 1])
    plt.xscale("log")
    # plt.ylim([0, 1])
    plt.yscale("log")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(file_name+'.png')        
    plt.savefig(file_name+'.pdf')        
    plt.savefig(file_name+'.eps')        
    plt.clf()    
    x=cv2.imread(file_name+'.png')[:,:,::-1]        
    return x
def tensor2im(input_image, imtype=np.uint8,mean=0.5,std=0.5,enhance=False):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable            
            image_tensor = input_image.data
        else:
            return input_image 
        if mean is None:
            mean = 0.5
        if std is None:
            std = 0.5       
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        image_numpy = (image_numpy+ (mean/std))* std  * 255.0  # post-processing: tranpose and scaling
        image_numpy = np.clip(image_numpy, a_min = 0, a_max = 255) 
        image_numpy = image_numpy.astype(imtype)        
        if enhance:
            image_numpy = np.tile(enhance_img(image_numpy[0,:,:]), (3, 1, 1))       
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))       
        
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)
def get_stat(stat,index,tile=False):
    """
    Return the statistic of a channel/band given its index. For example, if we have 3 infra bands (1050, 1200, 1300) and we want the mean of the 
    1050nm infra bands we would pass an array containing all the means of all the bands and and index of 0 would mean we want the first band/channel which is 1050nm.
    This is used with tensorboard to denormalize the infra channels for visualization.
    """
    if stat is None:
        return 0.5
    tmp=None
    if tile:
        tmp=np.tile(stat[index,:,:],(1,1,1))                           
    else:
        tmp=stat[index,:,:]
    return tmp
    
def scale_img(infra):
    """
        Convert a 16-bit image to 8-bit (usually infra images are 16-bit)
    """
    return cv2.normalize(infra,None,255,0,cv2.NORM_MINMAX,cv2.CV_8UC1)

def enhance_img(infra,scale=False):
    """
    Return a contrast enhanced version of an image. We use it with 16-bit infra images which will return infra images with the same contrast enhancement regardless of the infra band
    or the contrast enhancement parameters (e.g., clipLimit).
    """
    #Contrast Limited Adaptive Histogram Equalization: https://docs.opencv.org/3.2.0/d5/daf/tutorial_py_histogram_equalization.html
    clahe = cv2.createCLAHE(clipLimit=0.01,tileGridSize=(8,8)) # for 16 bit images, the clipLimit parameter doesn't change the contrast, however if the image is converted first to 8 bit the clipLimit controls the contrast. For now, using the 16-bit images produces consistent contrast for different images
    infra = clahe.apply(infra) 
    if scale:
        infra=scale_img(infra)       
    return infra
    
def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    return mean    


def save_image(image_numpy, image_path):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
