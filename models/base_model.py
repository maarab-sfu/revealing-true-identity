import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from . import networks
import numpy as np

class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this fucntion, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         specify the images that you want to display and save.
            -- self.visual_names (str list):        define networks used in our training.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir
        if opt.preprocess != 'scale_width':  # with [scale_width], input images might have different sizes, which hurts the performance of cudnn.benchmark.
            torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        self.infra_mean = np.zeros((3,self.opt.crop_size,self.opt.crop_size))
        self.infra_std  = np.zeros((3,self.opt.crop_size,self.opt.crop_size))    
        self.rgb_mean = np.zeros((3,self.opt.crop_size,self.opt.crop_size))
        self.rgb_std  = np.zeros((3,self.opt.crop_size,self.opt.crop_size))
        self.metric = None # used for learning rate policy 'plateau'
        self.epoch = 0

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass
    
    def set_epoch(self, epoch):
        self.epoch = epoch
        
    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def setup(self, opt, load_path=None):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if not self.isTrain or opt.continue_train:
            load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            self.load_networks(load_suffix,load_path)
        self.print_networks(opt.verbose)

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()
                
    def train(self):
        """Make models train mode during training time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.train()

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            self.compute_visuals()

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def get_image_paths(self):
        """ Return image paths that are used to load current data"""
        return self.image_paths

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for scheduler in self.schedulers:
            scheduler.step(self.metric)
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    def save_networks(self, epoch, temp_path=None):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = []
                if temp_path is not None:
                    save_path = os.path.join(temp_path, save_filename)
                else:
                    save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)
    def updateNet(self,load_state_dict,net_state_dict,idx):
        sz_idx=1
        if idx <0:
            sz_idx=0
        size_saved     = load_state_dict[list(load_state_dict)[idx]].size()[sz_idx] # e.g., model was trained on RGB (3 channels)                
        size_current   = net_state_dict[list(net_state_dict)[idx]].size()[sz_idx] # e.g., current model uses RGB + 3 SWIR + 1 NIR (7 channels)
        rem            = size_current % size_saved # 7 % 3 so remaining will be 1                
        saved_weights  = load_state_dict[list(load_state_dict)[idx]]
        print('size saved:{}, size current:{}, rem:{}, idx:{}'.format(size_saved,size_current,rem,idx))
        if idx <0: # last layer
            if len(load_state_dict[list(load_state_dict)[idx]].size())==4: # weights
                if rem > 0: #number of times to replicate is not divisble by 3                
                    net_state_dict[list(net_state_dict)[idx]] = torch.cat([saved_weights.repeat((size_current-rem)//size_saved,1,1,1),saved_weights[0:rem,:,:,:]],dim=1) # replicate the saved weights (7-1)/3=2 times (i.e., RGBRGB) and take one remaining (i.e., RGBRGBR)
                else: #number of times to replicate is divisible by 3
                    net_state_dict[list(net_state_dict)[idx]] = saved_weights.repeat((size_current-rem)//size_saved,1,1,1) # replicate the saved weights (7-1)/3=2 times (i.e., RGBRGB) and take one remaining (i.e., RGBRGBR)
            else: #bias
                net_state_dict[list(net_state_dict)[idx]] = saved_weights.repeat((size_current-rem)//size_saved) # replicate the saved weights (7-1)/3=2 times (i.e., RGBRGB) and take one remaining (i.e., RGBRGBR)                
        else: #first layer
            net_state_dict[list(net_state_dict)[idx]] = torch.cat([saved_weights.repeat(1,(size_current-rem)//size_saved,1,1),saved_weights[:,0:rem,:,:]],dim=1) # replicate the saved weights (7-1)/3=2 times (i.e., RGBRGB) and take one remaining (i.e., RGBRGBR)        
        
    def load_networks(self, epoch, temp_path=None):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = None
                if temp_path is not None:
                    load_path = os.path.join(temp_path, load_filename)
                else:
                    load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                load_state_dict     = torch.load(load_path, map_location=str(self.device))
                net_state_dict = net.state_dict()
                print('first layer')
                size_saved     = load_state_dict[list(load_state_dict)[0]].size()[1] #number of input channel of the pretrained model
                size_current   = net_state_dict[list(net_state_dict)[0]].size()[1] # e.g., current model uses RGB + 3 SWIR + 1 NIR (7 channels)
                if size_current!=size_saved:
                    self.updateNet(load_state_dict,net_state_dict,0) #replicate weights for input                 
                    if hasattr(net,'conc') and net.conc: # this means we have two inputs that are concatenated inside the network
                        print('first layer conc')                    
                        idx=np.inf
                        for i in range(1,len(list(load_state_dict))): # we search for a conv layer with the same number of input channels as the first conv layer
                            if len(load_state_dict[list(load_state_dict)[i]].size()) == 4 and size_saved == load_state_dict[list(load_state_dict)[i]].size()[1]:
                                idx=i
                                break
                        self.updateNet(load_state_dict,net_state_dict,idx) #replicate weights for input in case it is to be concatenated
                        load_state_dict.pop(list(load_state_dict)[idx], None) # remove the weights of the first layer of the second input since we set it manually
                    load_state_dict.pop(list(load_state_dict)[0], None) # remove the weights of the first layer (containing input channel) since we set it manually
                    if self.opt.model=='cycle_gan' and '_G_' in load_path: # replicate weights for output of generators too
                        idx = -1 if 'weight' in list(load_state_dict)[-1] else -2
                        print('last layer')
                        self.updateNet(load_state_dict,net_state_dict,idx)
                        if idx==-2: #means last conv layer had bias which needs to be adjusted according to the number of channels
                            self.updateNet(load_state_dict,net_state_dict,-1)
                            load_state_dict.pop(list(load_state_dict)[-1], None) # remove the weights of the last layer since we set it manually                
                            idx=-1
                        load_state_dict.pop(list(load_state_dict)[idx], None) # remove the weights of the last layer since we set it manually                
                if hasattr(load_state_dict, '_metadata'):
                    del load_state_dict._metadata                                                
                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(load_state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(load_state_dict, net, key.split('.'))

                # strict is set to false to be able to load whatever layers that exist in the saved model (because we removed the weights of the first layer from the loaded model since we replicate them manually to handle the difference in the number of channels)                    
                net.load_state_dict(load_state_dict, strict=False) 

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
