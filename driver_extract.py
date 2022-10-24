import argparse
import time
import argparse
import imutils
from PIL import Image
import random
import imageio
import numpy as np
import os.path
import scipy.misc
from data.image_folder import make_dataset
import cv2
import traceback
import logging
from batl_utils_modules.dataIO.odin_h5_dataset import *
import warnings
import csv
from collections import OrderedDict
from face_data_extractor.face_data_extractor import *
from batl_isi_pad_modules.self_test_attributes import * 
from batl_isi_pad_modules.utils.ImageTransformer import ImageTransformer
MAKEUP_ATTACK = "m50110004"
BONA_FIDE = "m00000000"

def set_ground_truth(readers,ground_truth_files):
    """
    Read the ground truth file of a dataset and set the ground truth label of the sample reader to attack or bona fide based on the desired criterion.    
    """
    mapping=OrderedDict()
    ## Some of the images in GCT3 have wrong collection ID, ST4. It can cause error while trying to extract image. You can use the "ds_name" below.
    # ds_name = "gct3"
    for key,value in ground_truth_files.items():
        
        with open(value, "r") as csv_file:
            rows = csv.DictReader(csv_file)
            for row in rows:
                label=-1
                if row['ground_truth'].startswith(BONA_FIDE):
                    label=0
                elif row['ground_truth'].startswith(MAKEUP_ATTACK):
                    # else:
                    label=1
                mapping[key+'-'+row['transaction_id']+'-'+row['trial_id']]=label    
    for _, sample_reader in enumerate(readers):        
        sample_reader.ground_truth_classifier= mapping[sample_reader.get_collection_id().lower()+'-'+sample_reader.transaction_id+'-'+sample_reader.trial_id]
        # sample_reader.ground_truth_classifier= mapping[ds_name+'-'+sample_reader.transaction_id+'-'+sample_reader.trial_id]
   
## Here you should insert the path to the 1. Face images, 2. the groundtruth file, and 3. the partition file.
choices=[
    [['/nas/vista-ssd01/batl/GCT-3/FACE'],['/nas/vista-ssd01/batl/GCT-3/batl_gct3_partitions/gct3_1025_1115_FACE_ground_truth.csv'],['/nas/vista-ssd01/batl/GCT-3/batl_gct3_partitions/gct3_dataset_partition_FACE_all_test.csv']],
    [['/nas/vista-ssd02/users/jmathai/batl/odin-phase-2-self-test-3-compressed/FACE_compressed'],['/nas/vista-ssd01/batl/SFU/makeup_label/st3_0225_0308_ground_truth.csv'],['/nas/vista-ssd01/batl/odin-phase-2-self-test-3/batl_st3_partitions/st3_dataset_partition_FACE_3folds.csv']],
    [['/nas/vista-ssd01/batl/odin-phase-2-self-test-4/FACE'],['/nas/vista-ssd01/batl/odin-phase-2-self-test-4/batl_st4_partitions/st4_0903_0910_FACE_ground_truth.csv'],['/nas/vista-ssd01/batl/odin-phase-2-self-test-4/batl_st4_partitions/st4_dataset_partition_FACE_3folds.csv']],
    [['/nas/vista-ssd02/users/jmathai/batl/odin-phase-2-self-test-3-compressed/FACE_compressed','/nas/vista-ssd01/batl/odin-phase-2-self-test-4/FACE'] , ['/nas/vista-ssd01/batl/SFU/makeup_label/st3_0225_0308_ground_truth.csv', '/nas/vista-ssd01/batl/odin-phase-2-self-test-4/batl_st4_partitions/st4_0903_0910_FACE_ground_truth.csv'], ['/nas/vista-ssd01/batl/odin-phase-2-self-test-3/batl_st3_partitions/st3_dataset_partition_FACE_3folds.csv','/nas/vista-ssd01/batl/odin-phase-2-self-test-4/batl_st4_partitions/st4_dataset_partition_FACE_3folds.csv']]]
paths=[]
if args.data.lower()=='st3':
    paths=choices[0]
elif args.data.lower()=='st4':
    paths=choices[1]
elif args.data.lower()=='gct3':
    paths=choices[2]
elif args.data.lower()=='combined':
    paths=choices[3]
folder_name =args.data.upper()

parser = argparse.ArgumentParser(description='Arguments to retrieve infra/RGB data.')
parser.add_argument('-dbd', dest='database_directories',
            default=paths[0],
            help='Path to the directory containing hdf5 files of the target dataset')
parser.add_argument('-gt', dest='ground_truth_files',
            default=paths[1],
            help='Path to the csv file containing ground truth for the dataset')
parser.add_argument('-dbp', dest='dataset_partitions_files',
            default=paths[2],
            help='Path to the csv file containing dataset partitioning into train, test, and validation')  
parser.add_argument("-p", "--shape-predictor", default="shape_predictor_68_face_landmarks.dat",help="path to facial landmark predictor")

args = parser.parse_args()

L = []
for i, dbd in enumerate(args.database_directories):
        L.append(FaceRGBInfraOdinH5Dataset(dbd, args.ground_truth_files[i], args.dataset_partitions_files[i]))
dataset = Odin5MultiDataset(L)
# partition = dataset.get_partition('test')
partition= dataset.get_partition('fold0')
partition+= dataset.get_partition('fold1')
partition+= dataset.get_partition('fold2')


ground_truth={}
for g in paths[1]:
    ground_truth[os.path.basename(g).split('_')[0]]=g

set_ground_truth(partition,ground_truth)                
names=['swir-1050nm', 'swir-1200nm','swir-1450nm']
labell = ""
for index,sample_reader in enumerate(partition):
    print(sample_reader.get_collection_id().lower()+'-'+sample_reader.transaction_id+'-'+sample_reader.trial_id)
    if sample_reader.ground_truth_classifier == -1:
        continue
    if sample_reader.ground_truth_classifier == 1:
        labell = "Attack"
    if sample_reader.ground_truth_classifier == 0:
        labell = "BonaFide"

    sample_ground_truth = sample_reader.get_ground_truth()
    sample_modality = sample_reader.modality.upper()

    print("===========================================" +
                    "=======================================================")
    print("Sample # {}:- Tx Id: {}, Tr Id: {} , Is PA?: {}, Modality: {}, PAI Code: {}".format(
            index, sample_reader.transaction_id,
            sample_reader.trial_id, sample_ground_truth, sample_modality,(sample_reader.get_reader_info_dict())['ground_truth']))	   
    # time_before = time.time()     
    sample_id="{}-{}-{}.h5".format(sample_reader.get_collection_id(),sample_reader.transaction_id, sample_reader.trial_id)               
    data = sample_reader.get_data()
    # bona_data = bona_sample.get_data()
    if data is None:
            print('face not found:{}'.format(sample_id))
            continue
    data=data[0]
    # bona_data = bona_data[0]
    print(type(data))
    rgb = data[:,:,0:3]
    # rgb_bona = bona_data[:,:,0:3]
    # rgb_total = cv2.hconcat([rgb, rgb_bona])
    swir = data[:,:,3:6]
    # swir_bona = bona_data[:,:,3:]
    # swir_total = cv2.hconcat([swir, swir_bona])
    # time_after = time.time()
    # print("data time:{}ms".format((time_after-time_before)*(10 ** 3)))        
    # print(np.shape(rgb))
    # print(type(rgb))
    # outimage = np.concatenate((rgb_total, swir_total), axis = 2)
    imageio.imwrite(folder_name+'/'+sample_reader.get_collection_id().lower()+'-'+labell+'_'+sample_reader.trial_id+'.png', rgb)
    # np.save(sample_reader.trial_id, outimage)


    for i in range(len(names)):         
            imageio.imwrite(folder_name+'/'+sample_reader.get_collection_id().lower()+'-'+labell+'_'+sample_reader.trial_id+'_'+names[i]+'.png', swir[:,:,i])