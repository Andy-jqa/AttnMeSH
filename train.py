__author__ = 'Qiao Jin'

from model.model import masked_BCE_loss
from model.model import mif_loss_th
from model import model
import numpy as np
import json
import random
import time
import torch
from torch import nn
from utils import utils

def main(params):

    num_epoch = params['num_epoch'] 
    batch_size = params['batch_size']
    lr = params['lr']

    logging_frequency = int(100 / batch_size)
    validate_frequency = int(10 * logging_frequency)

    use_cuda = torch.cuda.is_available()

    # Model initialization
    print('Initialize the model...')
    attnmesh = model.AttentionMeSH(params['w2v_dir'])
    print('Model initialized!')
    if use_cuda: attnmesh.cuda() 

    num_iter = 0
    t_start = time.time()

    optimizer = torch.optim.Adam(attnmesh.parameters(),lr=lr)

    print('Preparing mini-batch loaders...')
    training_set = json.load(open('data/train.json'))
    test_set = json.load(open('data/test.json'))
    batch_loader_train = utils.MiniBatchLoader(training_set, batch_size, N_mask=256, whether_train=1)
    batch_loader_val = utils.MiniBatchLoader(test_set, batch_size, N_mask=256, whether_train=0)
    print('Mini-batch loaders ready!')

    for epoch in range(num_epoch): 
        # Data preparation

        for data_tuple in batch_loader_train:
            t_elapsed = time.time() - t_start

            mesh_mask = torch.Tensor(data_tuple[2]).type(torch.LongTensor) # B x Ntrain, indice var
            words = torch.Tensor(data_tuple[0]).type(torch.LongTensor) # B x L, indice var
            word_mask = torch.Tensor(data_tuple[3])

            if use_cuda:
                mesh_mask = mesh_mask.cuda()
                words = words.cuda()
                word_mask = word_mask.cuda()

            pred = attnmesh(words, word_mask, mesh_mask, train=True) # B x Ntrain, prob var
            label = torch.Tensor(data_tuple[1]) # B x Nm, {1,0} var

            if use_cuda: label = label.cuda()

            loss = masked_BCE_loss(pred, label, mesh_mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # log
            if num_iter % logging_frequency == 0:
                message = ('Epoch %d Iter %d TRAIN loss=%.4e elapsed=%.1f'  % (epoch, num_iter, loss.detach(), t_elapsed))
                print(message)
                

            # Evaluation
            if num_iter % validate_frequency == 0:

                sum_output = 0.
                sum_target = 0.
                sum_product = 0.

                for data_tuple in batch_loader_val:
                    mesh_mask = torch.Tensor(data_tuple[2]).type(torch.LongTensor) # B x Ntrain, indice var
                    words = torch.Tensor(data_tuple[0]).type(torch.LongTensor) # B x L, indice var
                    word_mask = torch.Tensor(data_tuple[3])

                    if use_cuda:
                        mesh_mask = mesh_mask.cuda()
                        words = words.cuda()
                        word_mask = word_mask.cuda()
                    
                    pred = attnmesh(words, word_mask, mesh_mask, train=False) # B x Npred, prob var
                    label = torch.Tensor(data_tuple[1]) # B x Nm, {1,0} var

                    if use_cuda: label = label.cuda()

                    loss = masked_BCE_loss(pred, label, mesh_mask)

                    mi = mif_loss_th(pred, label, mesh_mask)
                    sum_output += mi[3]
                    sum_target += mi[4]
                    sum_product += mi[5]

                mip = sum_product / sum_output
                mir = sum_product / sum_target
                mif = 2 * mip * mir / (mip + mir)
                
                message = ('Epoch %d EVAL loss=%.4e MiF=%.4f MiP=%.4f MiR=%.4f' % (epoch,loss.detach(),mif,mip,mir))
                print(message)

            num_iter += 1
