__author__ = 'Qiao Jin'

import math
import numpy as np
import os
import torch
from torch import nn

use_cuda = torch.cuda.is_available()

def index_to_onehot(batch_index, B, chain_size, index_size):
    batch_index = torch.unsqueeze(batch_index, dim=2)
    if use_cuda:
        return torch.zeros(B, chain_size, index_size).cuda().scatter_(2, batch_index.data, 1)
    else:
        return torch.zeros(B, chain_size, index_size).scatter_(2, batch_index.data, 1)

class AttentionMeSH(nn.Module):

    def __init__(self, w2v_dir):
        '''
        '''
        super(AttentionMeSH,self).__init__()

        print('Load word2vec and mesh embedding, this might take a while')
        wordemb = np.loadtxt(os.path.join(w2v_dir, 'vectors.txt'))
        meshemb = np.load('mesh_emb/mesh_emb.npy')
        print('Word2vec and mesh embedding loaded')

        Dmodel = 200
        self.Dm = Dmodel
        self.max_length = 512

        wordemb = np.concatenate([wordemb, np.mean(wordemb, axis=0, keepdims=True)],axis=0)

        self.word_emb = nn.Embedding(wordemb.shape[0], Dmodel)
        self.word_emb.weight = nn.Parameter(torch.Tensor(wordemb))

        self.mesh_emb = nn.Embedding(meshemb.shape[0], Dmodel)
        self.mesh_emb.weight = nn.Parameter(torch.Tensor(meshemb))

        self.linear = nn.Linear(Dmodel, 2*Dmodel)
        
        self.mesh_weight = nn.Embedding(meshemb.shape[0], Dmodel)
        self.mesh_weight.weight = nn.Parameter(torch.Tensor(meshemb))

        self.mesh_bias = nn.Embedding(meshemb.shape[0],1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.2)

        self.gru = nn.GRU(Dmodel,Dmodel,num_layers=3,batch_first=True,bidirectional=True)
        self.gru_init = nn.Parameter(torch.randn(3*2,1,Dmodel), requires_grad=True)

    def forward(self,word,word_mask,mesh_mask,train):
        '''
        word: B x L variable indices
        word_mask: B x L
        mesh_mask: B x Nmask
        '''
        B = word.size(0)
        L = min(word.size(1), 512)
        N_mask = mesh_mask.size(1)
        eps = 1e-15

        word = word[:,:512]
        word_mask = word_mask[:,:512]

        word = self.word_emb(word) # B x L x Dm
        word = word * word_mask.unsqueeze(dim=2)

        if train:
            word = self.dropout(word)

        if use_cuda: seq_lengths = torch.sum(word_mask,1).type(torch.cuda.LongTensor) # B
        else: seq_lengths = torch.sum(word_mask,1).type(torch.LongTensor) # B

        seq_lengths = list(seq_lengths.data) # B

        seq_lengths = np.array(seq_lengths)
        perm = np.argsort(-seq_lengths)
        perm_i = np.argsort(perm)
        seq_lengths = seq_lengths[perm]

        if use_cuda: word = word[torch.Tensor(perm).type(torch.cuda.LongTensor)] # permuted B x L x Dm
        else: word = word[torch.Tensor(perm).type(torch.LongTensor)]

        seq_lengths = np.where(seq_lengths <= 0, 1, seq_lengths)
        pack = nn.utils.rnn.pack_padded_sequence(word, seq_lengths, batch_first=True)

        init = self.gru_init.expand(-1,B,-1).contiguous()
        word = self.gru(pack,init)[0] # B x L x 2Dm 
        word = nn.utils.rnn.pad_packed_sequence(word, batch_first=True)[0] # B x L(?) x 2Dm

        if word.size(1) != L:
            if use_cuda: padding = torch.Tensor(torch.zeros(B,L-word.size(1),2*self.Dm)).type(torch.cuda.FloatTensor)
            else: padding = torch.Tensor(torch.zeros(B,L-word.size(1),2*self.Dm)).type(torch.FloatTensor)
            word = torch.cat([word,padding],dim=1)

        if use_cuda: word = word[torch.from_numpy(perm_i).type(torch.cuda.LongTensor)] # B x L x 2Dm
        else: word = word[torch.from_numpy(perm_i).type(torch.LongTensor)]  

        mesh = self.mesh_emb(mesh_mask) # B x Nmask x Dm
        mesh = self.linear(mesh) # B x Nmask x 2Dm
        
        if train:
            mesh = self.dropout(mesh)

        pre_attn = torch.bmm(mesh,word.permute(0,2,1))  # B x Nmask x L
        pre_attn[(word_mask.unsqueeze(dim=1) == 0).expand(B, N_mask, L)] = float('-inf')
        attn_max = torch.max(pre_attn, dim=2, keepdim=True)[0]
        attn_exp = torch.exp(pre_attn - attn_max)
        attn = attn_exp / (torch.sum(attn_exp, dim=2, keepdim=True) + eps)

        doc_repr = torch.bmm(attn,word) # B x Nmask x 2Dm

        mesh_weight = self.linear(self.mesh_weight(mesh_mask)) # B x Nmask x 2Dm
        mesh_bias = self.mesh_bias(mesh_mask).squeeze(dim=2) # B x Nmask

        output = torch.sum(doc_repr*mesh_weight,dim=2) # B x Nmask
        output = self.sigmoid(output + mesh_bias)
    
        return output


def masked_BCE_loss(output, target, mask):
    # target: B x Nm,
    # mask and output: B x Ntrain
    # weight: B
    eps = 1e-12
    B = output.size(0)
    N_train = mask.size(1)

    n_mask = index_to_onehot(mask, B, N_train, 28340) # B x Ntrain x Nm
    
    target = torch.bmm(n_mask, torch.unsqueeze(target,2)).squeeze() # B x Ntrain

    n = B * N_train 
    loss = -torch.sum((target * torch.log(output + eps) + (1 - target) * torch.log(1 - output + eps)))
    loss = loss / n

    return loss

def mif_loss_th(output, target, mask):
    p = 0.5
    eps = 1e-12
    B = output.size(0)
    N_train = mask.size(1)
    n_mask = index_to_onehot(mask, B, N_train, 28340) # B x Npred x Nm
    output = torch.bmm(n_mask.permute(0,2,1), torch.unsqueeze(output,2)).squeeze() # B x Nm
 
    if B == 1:
        output=torch.unsqueeze(output,0)
    output = output.data.cpu().numpy() # B x Nm array 
    output = (output>p).astype(np.int) # B x Nm array
    pred = np.nonzero(output[0])[0]

    target = target.data.cpu().numpy()
        
    product = output * target
    sum_product = np.sum(product)
    output_sum = np.sum(output)
    target_sum = np.sum(target)
    mip = ((sum_product / output_sum) + eps)
    mir = ((sum_product / target_sum) + eps)

    return (1-(2*mip*mir/(mip+mir)), mip, mir, output_sum, target_sum, sum_product, pred)


def predict(output, mask, p):
    eps = 1e-12
    B = output.size(0)
    N_train = mask.size(1)

    n_mask = index_to_onehot(mask, B, N_train, 28340) # B x Npred x Nm
    output = torch.bmm(n_mask.permute(0,2,1), torch.unsqueeze(output,2)).squeeze() # B x Nm
 
    if B == 1:
        output=torch.unsqueeze(output,0)
    output = output.data.cpu().numpy() # B x Nm array 
    output = (output>p).astype(np.int) # B x Nm array
    pred = np.nonzero(output[0])[0]

    return pred
