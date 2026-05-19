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

    state=2
    print('state is:', state)

    # PATH = 'D:/guohuayan/pytorch/FAS_fit/data/FAS_state'+str(state)+'_phi_all.mat'
    # test_in = mat73.loadmat(PATH)['test_in']
    # test_out = mat73.loadmat(PATH)['test_out']
    # test_in=torch.from_numpy(test_in)
    # test_in=test_in.float()
    # test_out=torch.from_numpy(test_out)
    # test_out=test_out.cfloat().contiguous().view(-1,1)
    # print(test_in.shape)
    # print(test_out.shape)

    # power_resp=test_out.abs().square().mean()
    # print(power_resp)
    
    # PATH = 'D:/guohuayan/pytorch/FAS_fit/data/FAS_state'+str(state)+'_phi_test_1e6.mat'
    # data_in = mat73.loadmat(PATH)['datain']
    # data_out = mat73.loadmat(PATH)['dataout']
    # data_in=torch.from_numpy(data_in)
    # train_in=data_in.float()
    # data_out=torch.from_numpy(data_out)
    # train_out=data_out.cfloat().contiguous().view(-1,1)
    # print(train_out.abs().square().mean())

    PATH = './data/FAS_state'+str(state)+'_phi_fit.pt'
    # PATH = 'D:/guohuayan/pytorch/IRS_fit/data/FAS_state'+str(state)+'_theta_fit.pt'
    train_in=torch.load(PATH)['train_in']
    train_out=torch.load(PATH)['train_out']
    val_in=torch.load(PATH)['val_in']
    val_out=torch.load(PATH)['val_out']
    power_resp=train_out.abs().square().mean()
    print(train_in.shape)
    print(val_in.shape)
    print(power_resp)
    print(val_out.abs().square().mean())
    print(torch.view_as_real(val_out).view(-1,2).square().sum(dim=(1)).mean())

    batch_size_train = 1024
    train_dataset =CustomImageDataset(angle_in=train_in, resp=train_out)
    trainloader = DataLoader(dataset=train_dataset, batch_size=batch_size_train, 
                                    shuffle=True, num_workers=2, pin_memory=True,drop_last=True)

    batch_size_test= 1024
    test_dataset =CustomImageDataset(angle_in=val_in, resp=val_out)
    testloader = DataLoader(dataset=test_dataset, batch_size=batch_size_test, 
                                    shuffle=False, num_workers=2, pin_memory=True,drop_last=True)
    


    fas_nn = FAS_resp_nn(dim_out=2,block=4,device=device)
    # fas_nn = FAS_resp_nn(dim_out=32,device=device)

    # pretrain_PATH ='D:/guohuayan/pytorch/FAS_fit/dnn_fas_fit_state_'+str(state)+'_phi_v1.pt'
    # state=torch.load(pretrain_PATH,map_location=device)
    # csi_enc.load_state_dict(state['enc'])
    # fas_nn.load_state_dict(state['fas_nn'])

    # pretrain_PATH ='D:/guohuayan/pytorch/FAS_fit/model/dnn_fas_fit_state_'+str(state)+'_phi_v1.pt'
    # fas_nn.load_state_dict(torch.load(pretrain_PATH))
    # print('load model')

    fas_nn.to(device)    

    learning_rate = 1e-3
    NUM_EPOCHS = 2000
    optimizer = optim.Adam(fas_nn.parameters(), lr=learning_rate, betas=(0.9, 0.999), 
                                eps=1e-08, weight_decay=1e-5, amsgrad=False)
    # optimizer = optim.SGD(fas_nn.parameters(), lr=0.1)
    # optimizer.load_state_dict(state['optimizer_state'])
    # scheduler.load_state_dict(state['scheduler_state'])
    # print('load optimizer')


    # Model_PATH ='D:/guohuayan/pytorch/FAS_fit/dnn_fas_fit_state_'+str(state)+'_theta_v2.pt'
    # logger = get_logger('D:/guohuayan/pytorch/FAS_fit/dnn_fas_fit_state_'+str(state)+'_theta_v2.log', 'info')

    Model_PATH ='./dnn_fas_fit_state_'+str(state)+'_phi_v2.pt'
    logger = get_logger('./dnn_fas_fit_state_'+str(state)+'_phi_v2.log', 'info')
    

    TEST_FREQUENCY = 1
    min_valid_loss = np.inf
    # min_mse = np.inf
    # the_last_loss = 0.0
    patience = 100
    trigger_times = 0
    for epoch in range(NUM_EPOCHS):
        mse_train = train(fas_nn,trainloader,mse_criterion,optimizer,device=device)
        mse_train = mse_train / power_resp
        # mse_train = train(fas_nn,trainloader,nmse_criterion,optimizer,device=device)
        
        logger.info("[epoch %03d]  mse loss: %.6f" % (epoch, mse_train))
        # print("[epoch %03d]  smooth loss: %.4f" % (epoch, smooth_final))

        if epoch % TEST_FREQUENCY == 0:
            # report test diagnostics
            test_loss =test_loop(fas_nn, testloader, mse_criterion, device=device)
            test_loss = test_loss  / power_resp
            # test_loss =test_loop(fas_nn, testloader, nmse_criterion, device=device)

            logger.info("[epoch %03d] test mse loss: %.6f" % (epoch,test_loss))

            logger.info("[epoch %03d] min mse loss: %.6f" % (epoch,min_valid_loss))

        if test_loss-min_valid_loss > 1e-10:
            trigger_times += 1
            # print('trigger times:', trigger_times)
            logger.info("trigger_times: %.1f" % trigger_times)
            if trigger_times >= patience:
                logger.info('Early stopping due to validation loss increase!\n')
                break
        else:
            # print('trigger times: 0')
            trigger_times = 0
        

        if min_valid_loss > test_loss:
            min_valid_loss = test_loss
            # Save
            # state = {'ris_nn':ris_nn.state_dict(),
            #         'optimizer_state': optimizer.state_dict()}
            # torch.save(state, Model_PATH)
            torch.save(fas_nn.state_dict(), Model_PATH)
            logger.info("model updated")
   