# ADD deeper resnet
import os
from pickle import TRUE

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.distributions import constraints
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
import torch.optim as optim
import  complexPyTorch.complexLayers as pnn
import  complexPyTorch.complexFunctions as pF
import math

# from complexNN.nn import cLinear

import torchvision
import matplotlib.pyplot as plt
from scipy.io import savemat

class CustomImageDataset(Dataset):
    def __init__(self, angle_in,  resp):
        self.angle_in=angle_in
        self.resp=resp
        self.len=self.resp.shape[0]
        # self.shape=H_label.shape
        # self.transform = transform
        # self.target_transform = target_transform

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        input_in = self.angle_in[idx]
        label = self.resp[idx]
        return input_in,  label
    
class Generate_feature(nn.Module):
    def __init__(self, dim_in=3,dim_out=4,device = torch.device('cpu')):
        super().__init__()

        self.dim_in=dim_in
        self.dim_out=dim_out
        self.dim_f=2*self.dim_out
        self.linear0 = nn.Sequential(
            nn.Linear(self.dim_in, self.dim_f),                  
            nn.Tanh(),                          
        )
        self.res1 = nn.Sequential(
            nn.Linear(self.dim_f, self.dim_f),                  
            nn.Tanh(), 
            nn.Linear(self.dim_f, self.dim_f),                  
            nn.Tanh(),                          
        )
        # self.res2 = nn.Sequential(
        #     nn.Linear(self.dim_f, self.dim_f),                  
        #     nn.Tanh(), 
        #     nn.Linear(self.dim_f, self.dim_f),                  
        #     nn.Tanh(),                          
        # )
        self.linear1 = nn.Sequential(
            nn.Linear(self.dim_f, self.dim_f),                  
            nn.Tanh(),                          
        )

        self.v = nn.Linear(self.dim_f, self.dim_f)
        self.k = nn.Linear(self.dim_f, self.dim_out)
        self.q = nn.Linear(self.dim_f, self.dim_out)

        self.to(device)

    def forward(self, x):
        x = self.linear0(x)
        out = self.res1(x)
        x = out+x
        # out = self.res2(x)
        # x = out+x
        x = self.linear1(x)
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        v = v.view(-1,self.dim_out,2)
        v = torch.view_as_complex(v)
        return q, k ,v
    
class multi_block_feature(nn.Module):
    def __init__(self, dim_in=3,dim_out=4,block=16, device = torch.device('cpu')):
        super().__init__()

        self.dim_in=dim_in
        self.dim_out=dim_out
        self.block = block
        self.device = device

        self.multi_block = nn.ModuleList([Generate_feature(dim_in=self.dim_in,dim_out=self.dim_out,device=self.device) for i in range(self.block)])

        self.to(device)

    def forward(self, x):
        batch_size=x.shape[0]
        q_w=torch.empty((batch_size,self.dim_out,self.block), dtype=torch.float,device=self.device)
        k_w=torch.empty((batch_size,self.dim_out,self.block), dtype=torch.float,device=self.device)
        v_w=torch.empty((batch_size,self.dim_out,self.block), dtype=torch.cfloat,device=self.device)
        for idx, m_block in enumerate(self.multi_block):
            q, k ,v = m_block(x)
            q_w[:,:,idx]=q
            k_w[:,:,idx]=k
            v_w[:,:,idx]=v

        return q_w, k_w, v_w
    
class attention(nn.Module):
    def __init__(self,  dim_out=4, device = torch.device('cpu')):
        super().__init__()

        self.dim_out=dim_out
        self.active=torch.nn.Softmax(dim=1)
        self.device = device
        self.to(device)

    def forward(self, q_w,k_w):
        q_w = q_w.permute(0,2,1)
        corr= torch.matmul(q_w, k_w)/math.sqrt(self.dim_out) # batch 16 16
        corr=self.active(corr)

        return corr

class final_out(nn.Module):
    def __init__(self, dim_out=4, device = torch.device('cpu')):
        super().__init__()

        self.dim_out=dim_out
        self.device = device

        self.decision = nn.Sequential(
            pnn.ComplexLinear(self.dim_out, self.dim_out),                  
            ComplexTanh(),  
            # pnn.ComplexReLU(),
            # pnn.ComplexLinear(self.dim_out, self.dim_out), 
            # pnn.ComplexTanh(),
            # ComplexSoftplus(),
            # pnn.ComplexReLU(),
            pnn.ComplexLinear(self.dim_out, 4),                   
            # ComplexSoftplus(),
            ComplexTanh(),
            pnn.ComplexLinear(4, 1),                              
        )

        self.to(device)

    def forward(self, x):
        out = self.decision(x)

        return out
    
class ComplexTanh(nn.Module):
    def __init__(self):
        super().__init__()
        self.r_softplus = nn.Tanh()
        self.i_softplus = nn.Tanh()

    # @staticmethod
    def forward(self, inp):
        return self.r_softplus(inp.real) + 1j*self.i_softplus(inp.imag)
    
class ComplexSoftplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.r_softplus = nn.Softplus()
        self.i_softplus = nn.Softplus()

    # @staticmethod
    def forward(self, inp):
        return self.r_softplus(inp.real) + 1j*self.i_softplus(inp.imag)

    
class FAS_resp_nn(nn.Module):
    # by default our latent space is 50-dimensional
    # and we use 400 hidden units
    # def __init__(self,  is_train=True,  device = torch.device('cpu')):
    def __init__(self, dim_in=3,dim_out=4,block=16,  device = torch.device('cpu')):
        super().__init__()
        # create the encoder and decoder networks
        self.dim_in=dim_in
        self.dim_out=dim_out
        self.block = block
        self.device = device
        # self.is_train=is_train

        self.feature = multi_block_feature(dim_in=self.dim_in,dim_out=self.dim_out,block=self.block,device=self.device)
        self.attention = attention(dim_out=self.dim_out,device=self.device)
        self.finalout = final_out(dim_out=self.dim_out,device=self.device)

        self.to(device)

    def forward(self, x):
        q_w, k_w, v_w =self.feature(x)
        corr =self.attention(q_w, k_w)
        corr= corr.to(torch.cfloat)

        weight_v=torch.matmul(v_w,corr).sum(dim=2)

        out = self.finalout(weight_v)

        return out 
