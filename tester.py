import os
import time
import torch
import datetime
import numpy as np

import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image

from sagan_models import Generator, Discriminator
from utils import *
import helper

import matplotlib.pyplot as plt

class Tester(object):
    def __init__(self, data_loader, config):
        
        self.data_loader = data_loader
        
        # Model hyper-parameters
        self.imsize = config.imsize
        self.g_num = config.g_num
        self.z_dim = config.z_dim
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.parallel = config.parallel

        self.lambda_gp = config.lambda_gp
        self.total_step = config.total_step
        self.d_iters = config.d_iters
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.lr_decay = config.lr_decay
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.pretrained_model = config.pretrained_model
        
        self.dataset = config.dataset
        self.use_tensorboard = config.use_tensorboard
        self.image_path = config.image_path
        self.log_path = config.log_path
        self.model_save_path = config.model_save_path
        self.sample_path = config.sample_path
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.version = config.version
        
        self.model_save_path = os.path.join(config.model_save_path, self.version)
        
        
        self.test_path = config.test_path
        self.test_path = os.path.join(config.test_path, self.version)
        
        
        self.build_model()
        
        self.load_pretrained_model()
        
        
    def build_model(self):

        self.G = Generator(self.batch_size,self.imsize, self.z_dim, self.g_conv_dim).cuda()
        self.D = Discriminator(self.batch_size,self.imsize, self.d_conv_dim).cuda()
        if self.parallel:
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)

        # Loss and optimizer
        # self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D.parameters()), self.d_lr, [self.beta1, self.beta2])
        
        
    def load_pretrained_model(self):
        self.G.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_G.pth'.format(self.pretrained_model))))
        self.D.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_D.pth'.format(self.pretrained_model))))
        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

        
    def test(self):
        
        num_of_images=9
        
        for i in range(500):
            z = tensor2var(torch.randn(num_of_images, self.z_dim))
            
            
            fake_images,_,_= self.G(z)
            
            #print(fake_images.data)
            #(9,3,w,h) -> (9,w,h,3)
            #print(fake_images.data.shape)
            #print(fake_images.data[0])
            
            #save_image(denorm(fake_images.data),
            #           os.path.join(self.test_path, '{}_fake.png'.format(i+1)),
            #           nrow=3)
            
            #file_name = os.path.join(self.test_path, '{}_fake_class_format.png'.format(i+1))
            
            #self.output_fig(denorm(fake_images.data), file_name)
            
            transpose_image = np.transpose(var2numpy(fake_images.data), (0, 2, 3, 1))            
            self.output_fig(transpose_image,
                            os.path.join(self.test_path, '{}_fake_class_format.png'.format(i+1)))
            
            
    def output_fig(self, images_array, file_name):
        # the shape of your images_array should be (9, width, height, 3),  28 <= width, height <= 112 
        plt.figure(figsize=(6, 6), dpi=100)
        plt.imshow(helper.images_square_grid(images_array))
        plt.axis("off")
        plt.savefig(file_name, bbox_inches='tight', pad_inches=0)