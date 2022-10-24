from util.codes import *
import torch
import numpy as np
import math
import torch.nn as nn
from torch.nn import init
import functools
import torchvision
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import models
import random


###############################################################################
# Helper Functions
###############################################################################
def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def gaussian(ins, epoch, opt, mean=0):    
    stddev = opt.stddev *(1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1))
    noise = Variable(ins.data.new(ins.size()).normal_(mean, stddev))
    return ins + noise    


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_C(input_nc, filters,  norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a classifer

    Parameters:
        input_nc (int) -- the number of channels in input images        
        filters (int) -- the number of filters in the first conv layer
        norm (str) -- the name of normalization layers used in the network: batch | instance | none        
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a classifier 
    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """        
    net = Classifier(input_nc,filters)    
    return init_net(net, init_type, init_gain, gpu_ids)



class Classifier(nn.Module):
    """Defines a Classifier"""

    def __init__(self,input_nc,filters):
        """Construct a Classifier

        Parameters:
            input_nc (int)  -- the number of channels in input images
            filters  (int)       -- the number of filters in the first conv layer                        
        """
        super(Classifier, self).__init__()        
        #BN before ReLu: https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout
        #increase filter as we go deeper: https://www.quora.com/Why-does-convolution-neutral-network-increment-the-number-of-filters-in-each-layer
        norm_layer = nn.BatchNorm2d
        kw = 3
        padw = 1  # ensure input size is same as output size
        sequence = [nn.Conv2d(input_nc, filters, kernel_size=kw, stride=1, padding=padw), norm_layer(filters),
                    nn.LeakyReLU(0.2, True), nn.Dropout(p=0.2)]
        sequence += [nn.MaxPool2d(kernel_size=2, stride=2)]
        sequence += [nn.Conv2d(filters, int(filters * 2), kernel_size=kw, stride=1, padding=padw),
                     norm_layer(int(filters * 2)), nn.LeakyReLU(0.2, True), nn.Dropout(p=0.2)]
        sequence += [nn.Conv2d(filters * 2, int(filters * 2), kernel_size=kw, stride=1, padding=padw),
                     norm_layer(int(filters * 2)), nn.LeakyReLU(0.2, True), nn.Dropout(p=0.2)]
        sequence += [nn.MaxPool2d(kernel_size=2, stride=2)]
        sequence += [Flatten(), nn.Dropout(p=0.5), nn.Linear(2048, 1024), nn.Sigmoid()]
        sequence += [nn.Dropout(p=0.2), nn.Linear(1024, 128), nn.Sigmoid()]
        sequence += [nn.Dropout(p=0.2), nn.Linear(128, 1), nn.Sigmoid()]
        self.model = nn.Sequential(*sequence)


        # norm_layer = nn.BatchNorm2d
        # model = [nn.ReflectionPad2d(3),
        #          nn.Conv2d(input_nc, filters, kernel_size=7, padding=0, bias=False),
        #          norm_layer(filters),
        #          nn.ReLU(True)]        
        # n_downsampling = 2
        # for i in range(n_downsampling):  # add downsampling layers
        #     mult = 2 ** i
        #     model += [nn.Conv2d(filters * mult, filters * mult * 2, kernel_size=3, stride=2, padding=1, bias=False),
        #               norm_layer(filters * mult * 2),
        #               nn.ReLU(True)]
        # n_blocks=2
        # mult = 2 ** n_downsampling
        # for i in range(n_blocks):       # add ResNet blocks
        #     model += [ResnetBlock(filters * mult, padding_type='reflect', norm_layer=norm_layer, use_dropout=True, use_bias=False)]
        # model += [Flatten(), nn.Linear(4096, 2)]
        # self.model = nn.Sequential(*model)

        # self.model = models.densenet121(pretrained=True)
        # # for param in self.model.parameters():
        # #     param.requires_grad=False
        # num_ftrs = self.model.classifier.in_features
        #self.model.classifier = nn.Linear(num_ftrs, 2) 
       
               

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

class Flatten(nn.Module):
    def forward(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s                
        return x.view(-1, num_features)




class ClassifierLoss(nn.Module):
    """Define different GAN objectives.

    The ClassifierLoss class.
    """

    def __init__(self):      
        super(ClassifierLoss, self).__init__()        
        self.loss = nn.BCELoss()          

    def __call__(self,prediction,label):             
        loss = self.loss(prediction, label)         
        return loss

def symmetry_loss(fake_data,real_data):      
    # for i in range(fake_data.size(0)):
    #        torchvision.utils.save_image(fake_data[i, :, :, :], 'before_flip{}.png'.format(i))
    f = torch.randn(fake_data.size(0),3,real_data.size(2),fake_data.size(3)).cuda()  
    sz = list(f.size())        
    for i in range(real_data.size(0)):
        for j in range(int(sz[3]/2)):            
            f[i,:,:,j]=fake_data[i,:,:,sz[3]-j-1]
            f[i,:,:,sz[3]-j-1]=fake_data[i,:,:,j]
    # for i in range(f.size(0)):
    #        torchvision.utils.save_image(f[i, :, :, :], 'flipped{}.png'.format(i))
    x= torch.div(torch.mean(torch.abs((f - fake_data))),2.0)
    # print(x)
    return x

def edge_loss(fake_data,real_data):
    """inspired by:https://discuss.pytorch.org/t/edge-loss-function-implementation/21136
    """
    tmp_fake = torch.randn(real_data.size(0),1,real_data.size(2),real_data.size(3)).cuda()
    tmp_real = torch.randn(real_data.size(0),1,real_data.size(2),real_data.size(3)).cuda()
    for i in range(real_data.size(0)):
        tmp_fake[i,:,:,:] = torch.mul(fake_data[i,0,:,:],0.2989)+torch.mul(fake_data[i,1,:,:],0.5870)+torch.mul(fake_data[i,2,:,:],0.1140)
        tmp_real[i,:,:,:] = torch.mul(real_data[i,0,:,:],0.2989)+torch.mul(real_data[i,1,:,:],0.5870)+torch.mul(real_data[i,2,:,:],0.1140)
        # print(real_data.size())
        # torchvision.utils.save_image(tmp_real[i, :, :, :], 'gray{}.png'.format(i))
    x_filter = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    y_filter = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    weights_x = x_filter.float().unsqueeze(0).unsqueeze(0).cuda()#unsequeeze two times to make the number of channels 1 and the batch size 1
    weights_y = y_filter.float().unsqueeze(0).unsqueeze(0).cuda()	

    g1_x = nn.functional.conv2d(tmp_real, weights_x)
    g2_x = nn.functional.conv2d(tmp_fake, weights_x)
    g1_y = nn.functional.conv2d(tmp_real, weights_y)
    g2_y = nn.functional.conv2d(tmp_fake, weights_y)
    
    g_1 = torch.sqrt(torch.pow(g1_x, 2) + torch.pow(g1_y, 2))
    g_2 = torch.sqrt(torch.pow(g2_x, 2) + torch.pow(g2_y, 2))
    # for i in range(g_1.size(0)):
    #        torchvision.utils.save_image(g_1[i, :, :, :], 'gradient{}.png'.format(i))
    x= torch.mean(torch.abs((g_1 - g_2)))
    # print(x)
    return x
    

def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[],conc=False):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9,use_deconvolution=True)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6,use_deconvolution=True)
    elif netG == 'resnet_6blocks+':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6,use_deconvolution=False)
    elif netG == 'drn':
        net = DilatedResnetGenerator(input_nc, output_nc, conc=conc)
    elif netG == 'unet_64':
        net = UnetGenerator(input_nc, output_nc, 6, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[], conc=False):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you cna specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    elif netD == 'dilated':     # classify if image is real or fake using dilated convolution
        net = DilatedDiscriminator(input_nc, ndf, conc=conc)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """
    #label smoothing?: https://towardsdatascience.com/gan-ways-to-improve-gan-performance-acf37f9f59b
    def __init__(self, opt, target_real_label=1.0,target_real_label_disc=0.95, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('real_label_disc', torch.tensor(target_real_label_disc)) # just an initial value
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = opt.gan_mode
        self.opt=opt
        if self.gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % self.gan_mode)

    def get_target_tensor(self, prediction, target_is_real,disc=False):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            if disc:
                target_real_label_disc=random.uniform(0.7, 1.2)
                self.register_buffer('real_label_disc', torch.tensor(target_real_label_disc))
                target_tensor = self.real_label_disc.cuda()	
            else:                
                target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real,disc=False):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.opt.netD == 'dilated':
            prediction=prediction[0]
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real,disc)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            alpha = alpha.to(device)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect',use_deconvolution=True):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            if use_deconvolution:
                model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                             kernel_size=3, stride=2,
                             padding=1, output_padding=1,
                             bias=use_bias)]
            else: #should reduce checkerboard artifacts and overall distortion https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/pull/382/files#diff-a56d77751e639c0dd4a453fc554f55db and https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/190
                  #When we actually tried it, the results were very blurry
                model += [nn.Upsample(scale_factor = 2, mode='nearest'),
                          nn.ReflectionPad2d(1),
                          nn.Conv2d(ngf * mult, int(ngf * mult / 2),kernel_size=3, stride=1, padding=0)]

            model += [norm_layer(int(ngf * mult / 2)),nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias, dilation=1):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias, dilation=dilation)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias, dilation=1):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(dilation)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(dilation)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias, dilation=dilation), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(dilation)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(dilation)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias, dilation=dilation), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""        
        out = x + self.conv_block(x)  # add skip connections        
        return out


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""
    #check: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/39
    #also: https://fomoro.com/research/article/receptive-field-calculator
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, conc=False):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
            conc            -- takes care of the case when we pass 2 images to the discriminator during the construction of the neural net and during the forward pass.
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        self.conc = conc
        ndf_first=ndf//2 if self.conc else ndf
        self.first = nn.Sequential(nn.Conv2d(input_nc, ndf_first, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True))
        if self.conc:
            self.first1 = nn.Sequential(nn.Conv2d(input_nc, ndf_first, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True))
        nf_mult = 1
        nf_mult_prev = 1
        sequence = []
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult      #1,2
            nf_mult = min(2 ** n, 8)    #2,4
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult              #4
        nf_mult = min(2 ** n_layers, 8)     #8
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input, input2=None):
        """Standard forward."""
        out=self.first(input)
        if self.conc:
            out2=self.first1(input2)
            out=torch.cat([out,out2],dim=1)        
        return self.model(out)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.InstanceNorm2d
        else:
            use_bias = norm_layer != nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


#http://openaccess.thecvf.com/content_ECCV_2018/papers/Aaron_Gokaslan_Improving_Shape_Deformation_ECCV_2018_paper.pdf
#https://github.com/brownvc/ganimorph/blob/master/model.py
class DilatedDiscriminator(nn.Module):
    """Defines a Dilated discriminator based on http://openaccess.thecvf.com/content_ECCV_2018/papers/Aaron_Gokaslan_Improving_Shape_Deformation_ECCV_2018_paper.pdf
        The architecture is slightly by removing the dilated block with a dilation factor of 8. This is because the images we're dealing with are 128x128, therefore the effective receptive field should be good enough with just 2 dilated blocks instead of 3.
    
    """

    def __init__(self, input_nc, ndf=64, conc=False):
        """Construct a Dilated discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            conc            -- takes care of the case when we pass 2 images to the discriminator during the construction of the neural net and during the forward pass.
        """       
        super(DilatedDiscriminator, self).__init__()  
        self.conc=conc    
        norm_layer=nn.InstanceNorm2d
        ndf_first=ndf//2 if self.conc else ndf
        self.relu1 =  nn.Sequential(
                      nn.Conv2d(input_nc, ndf_first, kernel_size=4, stride=2,padding=1), 
                      nn.LeakyReLU(0.2, True),
                      nn.Conv2d(ndf_first, ndf_first * 2, kernel_size=4, stride=2,padding=1),
                      norm_layer(ndf_first * 2), 
                      nn.LeakyReLU(0.2, True))
        if self.conc:
            self.relu11 =  nn.Sequential(
                      nn.Conv2d(input_nc, ndf_first, kernel_size=4, stride=2,padding=1), 
                      nn.LeakyReLU(0.2, True),
                      nn.Conv2d(ndf_first, ndf_first * 2, kernel_size=4, stride=2,padding=1),
                      norm_layer(ndf_first * 2), 
                      nn.LeakyReLU(0.2, True))            
                
        self.relu2 = nn.Sequential(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2,padding=1), norm_layer(ndf * 4), nn.LeakyReLU(0.2, True))
        self.relu3 = nn.Sequential(nn.Conv2d(ndf * 4, ndf * 4, kernel_size=3, stride=1,padding=1), norm_layer(ndf * 4), nn.LeakyReLU(0.2, True))
        self.atrous = nn.Sequential(nn.Conv2d(ndf * 4, ndf * 4, kernel_size=3, stride=1,dilation=2,padding=2), norm_layer(ndf * 4), nn.LeakyReLU(0.2, True))
        self.atrous2 = nn.Sequential(nn.Conv2d(ndf * 4, ndf * 4, kernel_size=3, stride=1,dilation=4,padding=4), norm_layer(ndf * 4), nn.LeakyReLU(0.2, True))
        self.atrous3 = nn.Sequential(nn.Conv2d(ndf * 4, ndf * 4, kernel_size=3, stride=1,dilation=8,padding=8), norm_layer(ndf * 4), nn.LeakyReLU(0.2, True))
        self.clean = nn.Sequential(nn.Conv2d(ndf * 8, ndf * 4, kernel_size=3, stride=1,padding=1), norm_layer(ndf * 4), nn.LeakyReLU(0.2, True))
        self.lsgan  = nn.Sequential(nn.Conv2d(ndf * 4, 1, kernel_size=4, stride=1,padding=1))
        self.flat = nn.Sequential(Flatten())
        self.fc = nn.Sequential(nn.Linear(32768, 2))

    def forward(self, input, input2=None):
        """Standard forward."""
        relu1=self.relu1(input)
        if self.conc:
            relu11=self.relu11(input2)
            relu1=torch.cat([relu1,relu11],dim=1)
        relu2=self.relu2(relu1)
        relu3=self.relu3(relu2)
        atrous=self.atrous(relu3)
        atrous2=self.atrous2(atrous)
        atrous3=self.atrous3(atrous2)
        skip=torch.cat([relu3,atrous3],dim=1)
        clean=self.clean(skip)
        lsgan=self.lsgan(clean)
        # score=self.fc(self.flat(clean))
        return lsgan,[relu1,relu2,relu3,atrous,atrous2,clean]



class DilatedResnetGenerator(nn.Module):
    """Dilated-Resnet-based generator that consists of Resnet blocks with dilated convolutions instead of down/up sampling.
        This generator architecture should eliminate the blocking artifacts that arise from upsampling (which is done through the deconvolution operation).

    We adapt PyTorch code and idea from Fisher Yu's DRN(https://github.com/fyu/drn/blob/master/drn.py)

    
    input_nc        -- the number of input channels
    output_nc       -- the number of output channels
    channels        -- the number of filters to use throughout the generator's network
    conc            -- takes care of the case when we pass 2 images to the discriminator during the construction of the neural net and during the forward pass.
    """    

    def __init__(self, input_nc, output_nc, channels=64, conc=False):
        super(DilatedResnetGenerator, self).__init__()
        self.conc = conc
        self.inplanes = channels  

        self.conv10 = nn.Conv2d(input_nc, channels//2 if self.conc else channels, kernel_size=7, stride=1, padding=3, bias=True) # gives channelsx128x128 or (channels/2)x128x128
        self.relu10 = nn.ReLU(inplace=True)

        if self.conc:
            self.conv11 = nn.Conv2d(input_nc, channels//2 , kernel_size=7, stride=1, padding=3, bias=True) # gives (channels/2)x128x128
            self.relu11 = nn.ReLU(inplace=True)
        n_blocks = 9
        dilation = np.repeat([2,4,8],n_blocks//3)        
        self.down = nn.Sequential(nn.Conv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1, bias=True),
                    nn.InstanceNorm2d(channels * 2),
                    nn.ReLU(True))
        self.blocks=[]
        for i in range(n_blocks):       # add ResNet blocks
            self.blocks += [ResnetBlock(channels * 2, padding_type='reflect', norm_layer=nn.InstanceNorm2d, use_dropout=False, use_bias=True, dilation= dilation[i])]
        self.blocks = nn.Sequential(*self.blocks)
        self.up = nn.ConvTranspose2d(channels * 2, channels,
                        kernel_size=3, stride=2,
                        padding=1, output_padding=1,
                        bias=True)
        
        #The original PairedCycleGAN paper used 3 blocks in the generator. However, since they used multiple generators for the different face parts, the 3 blocks were enough (i.e., the effective receptive field was acceptable).
        #However, since we are trying to generate low resolution face images with just one generator, and since a whole face image is more complex than a single part of the face, we increased the number of blocks from 3 to 6 (also the dilation factor is increased correspondigly). This should increase the effective receptive field to be more adequate with the complexity of the whole face image.

        # self.layer1 = self._make_layer(BasicBlock, channels,  1)     # gives channelsx128x128
        # self.layer2 = self._make_layer(BasicBlock, channels,  1, 2)  # gives channelsx128x128
        # self.layer3 = self._make_layer(BasicBlock, channels,  1, 4)  # gives channelsx128x128
        # self.layer4 = self._make_layer(BasicBlock, channels,  1, 8)  # gives channelsx128x128
        # self.layer5 = self._make_layer(BasicBlock, channels,  1, 16) # gives channelsx128x128

        self.conv2 = nn.Conv2d(channels, channels//2, kernel_size=3, stride=1, padding=2, dilation=(2,2), bias=False) # gives (channels/2)x128x128 
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(channels//2, output_nc, kernel_size=3, stride=1, padding=1, bias=False) # gives 6x128x128 
        self.tanh3 = nn.Tanh()                
        
            

    def _make_layer(self, block, planes,  stride=1, dilation=1, residual=True):
        assert dilation == 1 or dilation % 2 == 0
        layers = list()
        layers.append(block( self.inplanes, planes, stride, (dilation, dilation), residual=residual))
        return nn.Sequential(*layers)


    def forward(self, x1, x2=None):        


        x = self.conv10(x1)        
        x = self.relu10(x)

        if self.conc:
            x2 = self.conv11(x2)            
            x2 = self.relu11(x2)
            x  = torch.cat([x,x2],dim=1)

        x = self.down(x)        
        x = self.blocks(x)
        x = self.up(x)
        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        # x = self.layer5(x)        

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.tanh3(x)

        return x


def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=(1, 1), residual=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, padding=dilation[0], dilation=dilation[0])        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout()
        
        self.conv2 = conv3x3(planes, planes, padding=dilation[1], dilation=dilation[1])        
        self.stride = stride
        self.residual = residual

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)        
        out = self.conv2(out)        

        if self.residual:
            out += residual        

        return out