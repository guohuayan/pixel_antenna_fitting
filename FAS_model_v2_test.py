# training based on the no smooth version
import os
from pickle import TRUE

from fit_model_fun import *
# from fit_model_fc_fun import *

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.distributions import constraints
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import matplotlib.pyplot as plt
from scipy.io import savemat
import math
from config_log import get_logger
import mat73


def train(net,trainloader, criterion,optimizer,device = torch.device('cpu')):
    # initialize loss accumulator
    normalizer_train = len(trainloader)
    running_loss = 0.0
    # net.train()
    mse_loss = 0.0

    # do a training epoch over each mini-batch x returned
    # by the data loader
    for i, data in enumerate(trainloader, 0):
        net.train() # make sure the batch is in training model in very iteration
        # if on GPU put mini-batch into CUDA memory
        data_in = data[0].to(device, dtype=torch.float)
        resp = data[1].to(device, dtype=torch.cfloat)

        # do ELBO gradient and accumulate loss
        optimizer.zero_grad()

        outputs = net(data_in)
        # loss = criterion(outputs, obs, prior_cost)
        loss=criterion(outputs,resp,device=device)

        # loss=loss_mse

        loss.backward(retain_graph=False)
        optimizer.step()

        # print statistics
        running_loss += loss.item() 

        mse_loss +=loss.item() 

    # return epoch loss
    

    mse_final = mse_loss / normalizer_train

    return mse_final

def test_loop(model,dataloader,  loss_fn,device = torch.device('cpu')):
    num_batches = len(dataloader)
    test_loss = 0.0
    model.eval()

    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            angle_in = data[0].to(device, dtype=torch.float)
            resp = data[1].to(device, dtype=torch.cfloat)

            # outputs = model.reconstruct_img(inputs)
            outputs = model(angle_in)
            
            # outputs=post_NN_module(outputs,Fm,Fa)
            loss = loss_fn(outputs, resp)
            test_loss += loss.item()

    test_loss = test_loss/num_batches
    
    return test_loss


def nmse_criterion(output, label,device = torch.device('cpu')):
    # loss = torch.square(output - target).sum()
    # output=torch.view_as_real(output)  # 1000 96 64 2
    label=torch.view_as_real(label).view(-1,2)
    output=torch.view_as_real(output).view(-1,2)
    mse1 = torch.square(output - label).sum(dim=(1))
    mse2 = torch.square( label).sum(dim=(1))
    mse = torch.div(mse1,mse2).mean()  

    loss = mse
    return loss

def mse_criterion(output, label,device = torch.device('cpu')):
    # loss = torch.square(output - target).sum()
    # output=torch.view_as_real(output)  # 1000 96 64 2
    label=torch.view_as_real(label).view(-1,2)
    output=torch.view_as_real(output).view(-1,2)
    mse1 = torch.square(output - label).sum(dim=(1))
    mse = mse1.mean()
    # p_label=label.square().sum(dim=(1)).mean()

    # loss = mse/p_label
    loss = mse
    return loss

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print(device)
    print('Using device:', device)

    for i0 in range(12):
        state=i0+1
        print('state is:', state)

        PATH = './data/FAS_state'+str(state)+'_phi_fit.pt'
        # PATH = './data/FAS_state'+str(state)+'_theta_fit.pt'
        test_in=torch.load(PATH)['test_in']
        test_out=torch.load(PATH)['test_out']

        output=torch.view_as_real(test_out).view(-1,2)
        mse1 = torch.square(output).sum(dim=(1))
        power_resp = mse1.mean()

        batch_size_test= 1024
        test_dataset =CustomImageDataset(angle_in=test_in, resp=test_out)
        testloader = DataLoader(dataset=test_dataset, batch_size=batch_size_test, 
                                        shuffle=False, num_workers=2, pin_memory=True,drop_last=True)
        


        fas_nn = FAS_resp_nn(dim_out=2,block=4,device=device)

        Model_PATH ='./model/dnn_fas_fit_state_'+str(state)+'_phi_v2.pt'
        # Model_PATH ='./model/dnn_fas_fit_state_'+str(state)+'_theta_v2.pt'
        fas_nn.load_state_dict(torch.load(Model_PATH,map_location=device))
        # print('load model')

        fas_nn.to(device)    

        
        # test_loss =test_loop(fas_nn, testloader, nmse_criterion, device=device) # the small value may not need acc

        test_loss =test_loop(fas_nn, testloader, mse_criterion, device=device)
        test_loss=test_loss/power_resp
            
        print("test nmse loss: %.6f" % (test_loss))


   