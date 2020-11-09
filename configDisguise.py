# -*- coding: utf-8 -*-
from easydict import EasyDict as edict

def get_config():
    conf = edict()
    conf.batch_size = 16
    conf.lr = 0.0002
    conf.beta1 = 0.5
    conf.beta2 = 0.999
    conf.epochs = 2000
    conf.save_dir = './saved_modelDisguise'
    conf.root='./PEAL_data'
    conf.savefig='./PEAL'
    conf.file='./dataset/LoadPEAL200.txt'
    conf.np = 2
    conf.nz = 50
    conf.nd = 200
    conf.TrainTag=True
    return conf
