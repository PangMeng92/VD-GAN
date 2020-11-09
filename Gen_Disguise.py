# -*- coding: utf-8 -*-
from model import Discriminator,Generator
from readerDisguise import get_batch
import torch as t
from torch import nn, optim
from configDisguise import get_config
import numpy as np
from tensorboardX import SummaryWriter
from torchvision import transforms
import torchvision.utils as vutils
import visdom
import os
import numpy as np
from scipy import misc
import imageio

def one_hot(label,depth):
    ones = t.sparse.torch.eye(depth)
    return ones.index_select(0,label)

def generateDisguise(conf):
    vis = visdom.Visdom()
    G = Generator(conf.nz, 3).cuda()
    T=t.load('./saved_modelDisguise/H%4d.pth'%1600)
    G.load_state_dict(T['g_net_list'])
    G.eval()

    train_loader = get_batch(conf.root, conf.file, conf.batch_size)

    steps = 0
    for epoch in range(1,conf.epochs+1):
        print('%d epoch ...'%(epoch))
        for i, batch_data in enumerate(train_loader):
            batch_image = batch_data[0]
            batch_id_label = batch_data[1]-1
            batch_disguise_label = batch_data[2]
            batch_pro = batch_data[3]

            batch_ones_label = t.ones(conf.batch_size)  
            batch_zeros_label = t.zeros(conf.batch_size)

            fixed_noise = t.FloatTensor(
                np.random.uniform(-1, 1, (conf.batch_size, conf.nz)))

            disguise_code_label = t.zeros(conf.batch_size).long()
            disguise_code = one_hot(disguise_code_label, conf.np)  # Condition 


            #cuda
            batch_image, batch_id_label, batch_disguise_label, batch_ones_label, batch_zeros_label = \
                batch_image.cuda(), batch_id_label.cuda(), batch_disguise_label.cuda(), batch_ones_label.cuda(), batch_zeros_label.cuda()

            fixed_noise, disguise_code, disguise_code_label = \
                fixed_noise.cuda(), disguise_code.cuda(), disguise_code_label.cuda()


            generated = G(batch_image, disguise_code, fixed_noise)
            
            
            steps += 1
            if i % 1 == 0:
                # x = vutils.make_grid(generated, normalize=True, scale_each=True)
                # writer.add_image('Image', x, i)
                
#                generated = generated.cpu().data.numpy()/2+0.5
#                generated = np.squeeze(generated)
#                generated = generated.transpose(1, 2, 0) 
#                save_gen = '{}_gen_train'.format(conf.savefig)
#                filename_gen = os.path.join(save_gen, '{}.png'.format(str(i+1)))
#                imageio.imwrite(filename_gen, generated)


#                batch_image = batch_image.cpu().data.numpy()/2+0.5
#                batch_image = np.squeeze(batch_image)
#                batch_image = batch_image.transpose(1, 2, 0) 
#                save_ori = '{}_ori_train'.format(conf.savefig)
#                filename_ori = os.path.join(save_ori, '{}.png'.format(str(i+1)))
#                imageio.imwrite(filename_ori, batch_image)
#                
#                
#                batch_pro = batch_pro.cpu().data.numpy()/2+0.5
#                batch_pro = np.squeeze(batch_pro)
#                batch_pro = batch_pro.transpose(1, 2, 0) 
#                save_pro = '{}_pro_train'.format(conf.savefig)
#                filename_pro = os.path.join(save_pro, '{}.png'.format(str(i+1)))
#                imageio.imwrite(filename_pro, batch_pro)
              
                
                generated = generated.cpu().data.numpy()/2+0.5
                generated = np.squeeze(generated)
                generated = generated.transpose(1, 2, 0) 
                save_gen = '{}_gen_test'.format(conf.savefig)
                filename_gen = os.path.join(save_gen, '{}.png'.format(str(i+1)))
                imageio.imwrite(filename_gen, generated)
                
                misc.imsave(filename, generated)
                generated4 = (generated1+generated2)/2
                batch_image = batch_image.cpu().data.numpy()/2+0.5
                batch_image = np.squeeze(batch_image)
                batch_image = batch_image.transpose(1, 2, 0) 
                save_ori = '{}_ori_test'.format(conf.savefig)
                filename_ori = os.path.join(save_ori, '{}.png'.format(str(i+1)))
                imageio.imwrite(filename_ori, batch_image)
                
                
                batch_pro = batch_pro.cpu().data.numpy()/2+0.5
                batch_pro = np.squeeze(batch_pro)
                batch_pro = batch_pro.transpose(1, 2, 0) 
                save_pro = '{}_pro_test'.format(conf.savefig)
                filename_pro = os.path.join(save_pro, '{}.png'.format(str(i+1)))
                imageio.imwrite(filename_pro, batch_pro)
                
if __name__=='__main__':
    conf = get_config()
    print(conf)
    generateDisguise(conf) 
    

