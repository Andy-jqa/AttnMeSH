__author__ = 'Qiao Jin'

import argparse
import config
import numpy as np
import random
import train

defaults = config.default_param

parser = argparse.ArgumentParser(description='Run AttentionMeSH')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=defaults['batch_size'], help='batch size (Default: 8)')
parser.add_argument('--seed', dest='seed', type=int, default=defaults['seed'], help='random seed (Default: 0)')
parser.add_argument('--lr', dest='lr', type=float, default=defaults['lr'], help='Adam learning rate (Default: 0.001)')
parser.add_argument('--num_epoch', dest='num_epoch', type=int, default=defaults['num_epoch'], help='number of epochs to train (Default: 64)')
parser.add_argument('--w2v_dir', dest='w2v_dir', type=str, default=defaults['w2v_dir'], help='The path to pre-trained w2v directory (Default word2vecTools)')
parser.add_argument('--mask_size', dest='mask_size', type=int, default=defaults['mask_size'], help='Size of MeSH mask (Default: 256)')

args = parser.parse_args()
params = vars(args)

np.random.seed(params['seed'])
random.seed(params['seed'])

train.main(params)
