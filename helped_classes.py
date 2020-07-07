# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 19:37:15 2020

@author: Dns1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
# from vedis import Vedis


# def get_current_state(user_id):
#     with Vedis(db_file) as db:
#         try:
#             return db[user_id].decode()
#         except KeyError:
#             return States.S_START.value

# def set_state(user_id, value):
#     with Vedis(db_file) as db:
#         try:
#             db[user_id] = value
#             return True
#         except:
#             return False

class ContentLoss(nn.Module):
        def __init__(self, target,):
            super(ContentLoss, self).__init__()
            self.target = target.detach()#это константа. Убираем ее из дерева вычеслений
            self.loss = F.mse_loss(self.target, self.target )#to initialize with something

        def forward(self, input):
            self.loss = F.mse_loss(input, self.target)
            return input
        
class StyleLoss(nn.Module):
        def __init__(self, target_feature):
            super(StyleLoss, self).__init__()
            self.target = gram_matrix(target_feature).detach()
            self.loss = F.mse_loss(self.target, self.target)# to initialize with something
            
        def gram_matrix(input):
            batch_size , h, w, f_map_num = input.size()  # batch size(=1)
            # b=number of feature maps
            # (h,w)=dimensions of a feature map (N=h*w)
    
            features = input.view(batch_size * h, w * f_map_num)  # resise F_XL into \hat F_XL
    
            G = torch.mm(features, features.t())  # compute the gram product
    
            # we 'normalize' the values of the gram matrix
            # by dividing by the number of element in each feature maps.
            return G.div(batch_size * h * w * f_map_num)

        def forward(self, input):
            G = gram_matrix(input)
            self.loss = F.mse_loss(G, self.target)
            return input
        
        
class Normalization(nn.Module):
        def __init__(self, mean, std):
            super(Normalization, self).__init__()
            # .view the mean and std to make them [C x 1 x 1] so that they can
            # directly work with image Tensor of shape [B x C x H x W].
            # B is batch size. C is number of channels. H is height and W is width.
            self.mean = torch.tensor(mean).view(-1, 1, 1)
            self.std = torch.tensor(std).view(-1, 1, 1)

        def forward(self, img):
            # normalize img
            return (img - self.mean) / self.std
        
def gram_matrix(input):
        batch_size , h, w, f_map_num = input.size()  # batch size(=1)
        # b=number of feature maps
        # (h,w)=dimensions of a feature map (N=h*w)

        features = input.view(batch_size * h, w * f_map_num)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(batch_size * h * w * f_map_num)