__author__ = 'Qiao Jin'

import json
import math
import random
import numpy as np

class MiniBatchLoader():

    def __init__(self, dataset, batch_size, N_mask, whether_train):
        self.data_set = dataset
        self.batch_size = batch_size
        self.N = len(dataset)
        self.N_mask = N_mask
        self.num_mesh = 28340

        self.train = whether_train
        self.shuffle = self.train

        self.reset()
        
    def __iter__(self):
        return self

    def reset(self):
        self.ptr = 0
        
        # self.batch_pool is a list of list
        
        idx = [i for i in range(self.N)]
        num_word = [len(article['token']) for article in self.data_set]

        if self.shuffle:
            r = random.random()
            random.shuffle(idx, lambda:r)
            random.shuffle(num_word, lambda:r)
        
        num_word = np.asarray(num_word) # sort, if same, according to loc in original list
        num_word_rank = np.argsort(num_word) # idx of doc from the smallest

        self.batch_pool = []
        self.batch_pool_len = []

        for i in num_word_rank:
            if len(self.batch_pool) == 0:
                self.batch_pool.append([idx[i]])
                self.batch_pool_len.append(num_word[i])
                continue
            if len(self.batch_pool[-1]) >= self.batch_size or num_word[i] != self.batch_pool_len[-1]:
                self.batch_pool.append([idx[i]])
                self.batch_pool_len.append(num_word[i])
                continue
            self.batch_pool[-1].append(idx[i])

        if self.shuffle:
            r = random.random()
            random.shuffle(self.batch_pool, lambda:r)
            random.shuffle(self.batch_pool_len, lambda:r)

    def __next__(self):
        if self.ptr == len(self.batch_pool):
            self.reset()
            raise StopIteration()

        indices = self.batch_pool[self.ptr] # a list of indices
        B = len(indices)
        L = self.batch_pool_len[self.ptr]
        doc = np.zeros((B, L))
        mask = np.zeros((B, L))
        mesh = np.zeros((B, self.num_mesh))
        weight = np.zeros((B,))
        mesh_mask = np.zeros((B, self.N_mask))

        for i in range(B):
            doc[i] = np.array(self.data_set[indices[i]]['token'])
            mask[i][:self.data_set[indices[i]]['length']] = 1
            mesh[i] = self.idx2onehot(self.data_set[indices[i]]['mesh'])
            
            if self.train:
                mesh_mask[i] = self.get_train_mask(self.data_set[indices[i]]['neighbor'],self.data_set[indices[i]]['mesh'],self.N_mask)
            else:
                mesh_mask[i] = self.get_pred_mask(self.data_set[indices[i]]['neighbor'],self.N_mask)

        self.ptr += 1

        return (doc, mesh, mesh_mask, mask)

    def idx2onehot(self,indices):
        onehot = [0 for i in range(self.num_mesh)]
        for i in indices:
            if i < len(onehot):
                onehot[i] = 1
        return np.asarray(onehot)

    def get_train_mask(self,indices,mesh,N):
        '''
        N_pos = len(mesh)
        N_neg = N - N_pos
        mask_idx = mesh + [tu[0] for tu in indices[-N_neg:]]
        '''
        mask_idx = mesh + [tu for tu in indices[-N:] if tu not in mesh] # length range from N to N + mesh
        mask_idx = mask_idx[:N]

        idx = 0
        while len(mask_idx) < N:
            if idx not in mask_idx:
                mask_idx += [idx]
            idx += 1

        return np.array(mask_idx)

    def get_pred_mask(self,indices,N):
        mask_idx = [tu for tu in indices[-N:]]

        idx = 0
        while len(mask_idx) < N:
            if idx not in mask_idx:
                mask_idx += [idx]
            idx += 1

        return np.array(mask_idx)
