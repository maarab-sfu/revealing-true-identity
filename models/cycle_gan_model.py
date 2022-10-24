import torch
import torchvision
import torchvision.transforms as transforms
import pytorch_msssim
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import random
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import imutils
import numpy as np
from util.util import enhance_img,tensor2im,get_stat,get_image_tensor,get_image
from util.codes import *
from LightCNN.light_cnn import LightCNN_29Layers_v2
from PIL import Image


class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        if self.opt.netG == 'drn':
            self.loss_names = ['D_A', 'G_A', 'G_A_style', 'cycle_A','cycle_perceptual_A','FM_A', 'idt_A', 'D_B', 'G_B', 'cycle_B','cycle_perceptual_B','FM_B', 'idt_B','face_feature_A','face_feature_B', 'D_makeup', 'sparse']
        else:
            self.loss_names = ['D_A', 'G_A', 'cycle_A','cycle_perceptual_A','FM_A', 'idt_A', 'D_B', 'G_B', 'cycle_B','cycle_perceptual_B','FM_B', 'idt_B','face_feature_A','face_feature_B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            if self.opt.netG == 'drn':
                self.model_names = ['G_A', 'G_B', 'D_A', 'D_B', 'D_makeup']
            else:
                self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X) 
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, conc=True)     
        for name,param in self.netG_A.named_parameters():
            if param.requires_grad == True:                
                print("\t",name)   
        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            if self.opt.netG == 'drn':
                self.netD_makeup = networks.define_D(opt.input_nc, opt.ndf, opt.netD, 
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, conc=True)
            
            
        if self.opt.lambda_face_identity > 0.0:
            self.face_model = torch.nn.DataParallel(LightCNN_29Layers_v2(num_classes=80013)).cuda()                       
            self.face_model.eval()
            self.face_model.load_state_dict((torch.load('/nas/home/aminarab/batl-utils/LightCNN/LightCNN_29Layers_V2_checkpoint.pth'))['state_dict'])
            

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt).to(self.device)  # define GAN loss.            
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionStyle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionFaceFeature = torch.nn.L1Loss()
            self.criterionClassification = torch.nn.CrossEntropyLoss()
            self.criterionPerceptualCycle = pytorch_msssim.SSIM()            
            self.criterionFeatureMatch = torch.nn.L1Loss()
            self.target_A = torch.autograd.Variable(torch.zeros([self.opt.batch_size],dtype=torch.long)).to(self.device)
            self.target_B = torch.autograd.Variable(torch.ones([self.opt.batch_size],dtype=torch.long)).to(self.device)                   
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            #self.optimizer_G = adabound.AdaBound(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, final_lr=0.1)
            #self.optimizer_D = adabound.AdaBound(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, final_lr=0.1)
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            if self.opt.netG == 'drn':
                self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters(), self.netD_makeup.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            else:
                self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        #AtoB = self.opt.direction == 'AtoB'        
        self.real_A = input['A'].to(self.device)
        # print(type(self.real_A))
        # print(self.real_A.size())
        # im = self.real_A.cpu().numpy()
        # im = im[0]
        # print(type(im))
        # print(im.shape)
        # im = Image.fromarray(np.rollaxis(im, 0,3))
        # im.save('test_ta.png')
        if self.opt.isTrain:        
            self.real_B = input['B'].to(self.device)
            # self.warped = input['W'].to(self.device)
        if 'A_paths' in input.keys():
            self.A_paths = input['A_paths']
        if 'B_paths' in input.keys():
            self.B_paths = input['B_paths']
        
        

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""         
        self.fake_B = self.netG_A(self.real_A)   # G_A(A)   ===> makeup removal   
        if self.opt.netG != 'drn':
            self.rec_A = self.netG_B(self.fake_B)                
        if self.opt.isTrain:            
            if self.opt.netG == 'drn':
                self.fake_A = self.netG_B(self.real_B, self.real_A)   # G_B(B)  ===> apply makeup, we use addition as in PairedCycleGAN which assumes that the makeup layer should be delta values from the original image            
            else:
                self.fake_A = self.netG_B(self.real_B)
            self.rec_B  = self.netG_A(self.fake_A)   # G_A(G_B(B)) ===> remove makeup after applying it     
            if self.opt.netG == 'drn':
                self.rec_A = self.netG_B(self.fake_B, self.fake_A) #===> apply makeup on the makeup removed version of A using the stylized version of B that adopted the original style of A                       
        else:
            if self.opt.netG == 'drn':
                self.rec_A = self.netG_B(self.fake_B, self.real_A) # while testing CycleGAN we don't have fake_A if use DRN to re-apply the makeup using it so we use real_A instead, this is just to have something to print while testing. While training/validating we have fake_A since isTrain will be true

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)        
        if self.opt.netD == 'dilated':
            pred_real=pred_real[0]
        lbl_true = True
        prob = random.uniform(0, 1)
        #https://medium.com/@utk.is.here/keep-calm-and-train-a-gan-pitfalls-and-tips-on-training-generative-adversarial-networks-edd529764aa9
        if prob<self.opt.prob:
            lbl_true=False

        
        pred_real =  networks.gaussian(pred_real,self.epoch,self.opt)
        loss_D_real = self.criterionGAN(pred_real, lbl_true,disc=True)
        # Fake        
        pred_fake = netD(fake.detach())
        if self.opt.netD == 'dilated':
            pred_fake=pred_fake[0]
        pred_fake =  networks.gaussian(pred_fake,self.epoch,self.opt)
        loss_D_fake = self.criterionGAN(pred_fake, not(lbl_true),disc=True)        
                    
        # Combined loss and calculate gradients        
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_makeup_basic(self, netD, real, fake):
        """Calculate GAN loss for the makeup discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real makeup image
            fake (tensor array) -- generated makeup image

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        #warping is used as an approximation used in PairedCycleGAN to make the makeup discriminator be able to say that 2 faces have the same makeup style or not (it's a synthentized version of the no makeup image with the makeup applied on it)        
        pred_real = netD(real, self.warped)        
        if self.opt.netD == 'dilated':
            pred_real=pred_real[0]
        lbl_true = True
        prob = random.uniform(0, 1)
        if prob<self.opt.prob:
            lbl_true=False
        pred_real =  networks.gaussian(pred_real,self.epoch,self.opt)
        loss_D_real = self.criterionGAN(pred_real, lbl_true,disc=True)
        # Fake        
        pred_fake = netD(real, fake.detach())
        if self.opt.netD == 'dilated':
            pred_fake=pred_fake[0]
        pred_fake =  networks.gaussian(pred_fake,self.epoch,self.opt)
        loss_D_fake = self.criterionGAN(pred_fake, not(lbl_true),disc=True)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D
            

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_D_makeup(self):
        """Calculate GAN loss for discriminator D_B"""        
        self.loss_D_makeup = self.backward_D_makeup_basic(self.netD_makeup, self.real_A, self.fake_A)
        

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        perceptual_factor = self.opt.perceptual_factor
        lambda_feature = self.opt.lambda_feature
        lambda_face_identity = self.opt.lambda_face_identity
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        out_D_A = self.netD_A(self.fake_B)        
        self.loss_G_A = self.criterionGAN(out_D_A, True)

        # GAN loss D_B(G_B(B))
        out_D_B = self.netD_B(self.fake_A)        
        self.loss_G_B = self.criterionGAN(out_D_B, True)

        self.loss_G_A_style = 0
        if self.opt.netG == 'drn':
            # GAN loss D_makeup(G_B(B))
            out_D_makeup_A = self.netD_makeup(self.real_A, self.fake_A)
            self.loss_G_A_style = self.criterionGAN(out_D_makeup_A, True)
        # Forward cycle loss || G_B(G_A(A), A) - A|| #remove makeup from A, then apply it again using A
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B, A)) - B|| #Eq (3) in PairedCycleGAN ==> apply makeup on B using A and then remove it
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B 

        self.loss_sparse = self.criterionCycle(self.fake_A, self.real_B) * 0.0    #check page 5 in PairedCycleGAN
        # Style loss || G_B(G_A(A), G_B(B, A)) - A|| #apply makeup on the no-makeup version of A using the stylized version of B that took the original style of A and then compare it with original A which had makeup                    
        # Forward perceptual cycle loss (SSIM)
        self.loss_cycle_perceptual_A = (1-self.criterionPerceptualCycle(self.rec_A, self.real_A)) * lambda_A * perceptual_factor
        # Backward perceptual cycle loss (SSIM)
        self.loss_cycle_perceptual_B = (1-self.criterionPerceptualCycle(self.rec_B, self.real_B)) * lambda_B * perceptual_factor
        # Backward perceptual cycle loss (SSIM)        
            
        # combined loss and calculate gradients
        self.loss_FM_A = 0
        self.loss_FM_B = 0
        if self.opt.netD == 'dilated':
            out_D_A_real = self.netD_A(self.real_B)
            out_D_B_real = self.netD_B(self.real_A)                        
            for i in range(0,len(out_D_A_real[1])):
                self.loss_FM_A += self.criterionFeatureMatch(out_D_A[1][i], out_D_A_real[1][i].detach())
                self.loss_FM_B += self.criterionFeatureMatch(out_D_B[1][i], out_D_B_real[1][i].detach())
            self.loss_FM_A /= len(out_D_A_real[1])
            self.loss_FM_B /= len(out_D_B_real[1])
            self.loss_FM_A *= lambda_feature
            self.loss_FM_B *= lambda_feature
        self.loss_face_feature_A = 0        
        self.loss_face_feature_B = 0        
        if self.opt.lambda_face_identity > 0.0:       
            mean=self.rgb_mean
            std= self.rgb_std                        
            
            tmp_rec_A =get_image_tensor(self.rec_A,mean,std)
            tmp_real_A=get_image_tensor(self.real_A,mean,std)
            tmp_rec_B =get_image_tensor(self.rec_B,mean,std)
            tmp_real_B=get_image_tensor(self.real_B,mean,std)
                        
            rec_A_feat =self.face_model(tmp_rec_A[:,:,::2,::2])[1]               
            real_A_feat=self.face_model(tmp_real_A[:,:,::2,::2])[1]
            rec_B_feat =self.face_model(tmp_rec_B[:,:,::2,::2])[1]                
            real_B_feat=self.face_model(tmp_real_B[:,:,::2,::2])[1]   
            
            self.loss_face_feature_A = self.criterionFaceFeature(rec_A_feat, real_A_feat) * lambda_face_identity
            self.loss_face_feature_B = self.criterionFaceFeature(rec_B_feat, real_B_feat) * lambda_face_identity             
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_G_A_style + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_cycle_perceptual_A + self.loss_cycle_perceptual_B + self.loss_FM_A + self.loss_FM_B + self.loss_face_feature_A + self.loss_face_feature_B + self.loss_sparse
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        if self.opt.netG == 'drn':
            self.set_requires_grad([self.netD_A, self.netD_B, self.netD_makeup], False)  # Ds require no gradients when optimizing Gs
        else:
            self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        if self.opt.netG == 'drn':
            self.set_requires_grad([self.netD_A, self.netD_B, self.netD_makeup], True)
        else:
            self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        if self.opt.netG == 'drn':
            self.backward_D_makeup()      # calculate graidents for D_makeup
        self.optimizer_D.step()  # update D_A and D_B's weights
