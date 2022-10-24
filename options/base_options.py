import argparse
import os
from util import util
import torch
import models
import data


class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--dataroot', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--binary_classifier_dir', type=str, default='/ISI_binary/', help='models are saved here')
        parser.add_argument('--print_freq', type=int, default=25, help='frequency of showing training results on console')
        # model parameters
        parser.add_argument('--mask', type=str, default='diff',help='diff or ssim or const')
        parser.add_argument('--infra_bands', type=str, default='1200nm,1450nm',help='list of infra bands to be used')
        parser.add_argument('--mode',default="train",help='train or test')
        parser.add_argument('--data',default="ST3",help='GCT2 or ST3 or combined')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--model', type=str, default='binary', help='chooses which model to use. [cycle_gan | pix2pix | test | colorization| binary]')
        parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB or infra')        
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels: 3 for RGB or infra')
        parser.add_argument('--filters', type=int, default=16, help='# of filters in the first conv layer for the binary model')        
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        parser.add_argument('--netD', type=str, default='basic', help='specify discriminator architecture [basic | n_layers | pixel| dilated]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
        parser.add_argument('--netG', type=str, default='resnet_9blocks', help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
        parser.add_argument('--stddev', type=float, default=0.1, help='standard deviation of gaussian noise which is added to the discriminator''s input')
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')        
        parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
        parser.add_argument('--include_normal_makeup', action='store_true', help='include bona fide normal makeup in the domain of makeup while training CycleGAN')    
        parser.add_argument('--blur', action='store_true', help='blur infra bands')    
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        parser.add_argument('--prob', type=float, default=0.05, help='label switching probability for discriminator')
        parser.add_argument('--thresh_calc', action='store_true', help='use eval mode during test time.')
        # dataset parameters  
        parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')      
        parser.add_argument('--dataset_mode', type=str, default='h5', help='chooses how datasets are loaded. [unaligned | aligned | single | colorization | h5]')
        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--load_size', type=int, default=128, help='scale images to this size')
        parser.add_argument('--crop_size', type=int, default=128, help='then crop to this size')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML')
        parser.add_argument('--matcher_thresh', type=float, default=0.5, help='matcher threshold')
        parser.add_argument('--binary_thresh', type=float, default=0.5, help='matcher threshold')
        parser.add_argument('--attack_only', action='store_true', help='attack vs no makeup only')        
        # additional parameters
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam') #https://medium.freecodecamp.org/how-to-pick-the-best-learning-rate-for-your-machine-learning-project-9c28865039a8
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--load_iter', type=int, default='0', help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
        parser.add_argument('--niter', type=int, default=200, help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=800, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--perceptual_factor', type=float, default=0.0, help='factor of perceptual loss as compared to L1 loss')
        parser.add_argument('--lambda_identity', type=float, default=0.0, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
        parser.add_argument('--lambda_class', type=float, default=0.0, help='classification loss weight')
        parser.add_argument('--lambda_feature', type=float, default=0.0, help='feature matching weight')
        parser.add_argument('--lambda_face_identity', type=float, default=0.0, help='face identity weight')     
        parser.add_argument('--method', type=str, default='inpaint', help='which method to use for detecting attacks [inpaint | rgb]. inpaint means we use the mask computed from difference between infra before and after makeup removal and applied to rgb to be inpainted. rgb means we use the rgb image after makeup removal from CycleGAN and pass it directly to the matcher to be compared with the rgb image before makeup removal.')   
        parser.add_argument('--matcher', type=str, default='CNN', help='which matcher to use [CNN | VeriLook]')   
        parser.add_argument('--test_fold', type=int, default=5, help='choose the test fold in 5 fold cross-validation')
        parser.add_argument('--binary_test', type=int, default=0, help='Are we testing the binary classifier')
        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with new defaults

        # modify dataset-related parser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
