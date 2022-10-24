import torch
from .base_model import BaseModel
import torch.nn as nn
from . import networks


class BinaryModel(BaseModel):
    """ This class implements the classifier for makeup and bona fide classes
    
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        """                
        return parser

    def __init__(self, opt):
        """Initialize the binary model class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['C']        
        self.model_names = ['C']        
        self.netC = networks.define_C(opt.input_nc, opt.filters, opt.norm,
                                       opt.init_type, opt.init_gain, self.gpu_ids)
        
        if self.isTrain:            
            # define loss functions
            self.criterionClassifier = networks.ClassifierLoss().to(self.device)       
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            
            params_to_update = []
            for name,param in self.netC.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    print("\t",name)                        
            #self.optimizer_C = adabound.AdaBound(params_to_update, lr=opt.lr, final_lr=0.1)
            #self.optimizer_C = torch.optim.SGD(params_to_update, lr=opt.lr, momentum=0.9)
            self.optimizer_C = torch.optim.Adam(params_to_update, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_C)                        
        else: # set the requires_grad to False for the new layer if we are testing
            self.set_requires_grad(self.netC,requires_grad=False)


    def set_input(self, inp):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.        
        """        
        self.input = inp['A'].to(self.device)
        self.paths = inp['path']
        if 'label' in inp.keys():
            self.target = inp['label'].to(self.device).float()              

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.score = self.netC(self.input)  # since the network outputs a prob for each class we want get the max score index (0 or 1)

    def backward_C(self):
        """Calculate  loss for the classifier"""        
        self.loss_C = self.criterionClassifier(self.score,self.target)
        self.loss_C.backward()
   
    def optimize_parameters(self):
        self.optimizer_C.zero_grad()        
        self.forward()                                   
        self.backward_C()                   
        self.optimizer_C.step()             
