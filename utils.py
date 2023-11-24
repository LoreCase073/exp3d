import torch
import torch.nn.functional as F
import os
import argparse
import json
import numpy as np
import torch.nn as nn

def composite_loss(output, target, c1, c2):
    #position term
    loss1 = torch.mean((output - target)**2)
    #velocity term, difference between adjacent elements
    loss2 = torch.mean(((output[:,1:,:]-output[:,:-1,:])-(target[:,1:,:]-target[:,:-1,:]))**2)
    loss = c1 * loss1 + c2 * loss2
    return loss

def compute_distance(output, target):
    d = ((output[:,:,:,0]-target[:,:,:,0])**(2))+((output[:,:,:,1]-target[:,:,:,1])**(2))+((output[:,:,:,2]-target[:,:,:,2])**(2))
    d = torch.sqrt(d)
    d = torch.mean(d)
    return d