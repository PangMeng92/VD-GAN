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

def one_hot(label,depth):
    ones = t.sparse.torch.eye(depth)
    return ones.index_select(0,label)

def trainDisguise(conf):
#    vis = visdom.Visdom()
    train_loader = get_batch(conf.root, conf.file, conf.batch_size)    #xxxxxx
    D = Discriminator(conf.nd, conf.np, 3).cuda()
    G = Generator(conf.nz, 3).cuda()
    D.train()
    G.train()

    optimizer_D = optim.Adam(D.parameters(),
                             lr=conf.lr,betas=(conf.beta1,conf.beta2))
    optimizer_G = optim.Adam(G.parameters(), lr=conf.lr,
                                           betas=(conf.beta1, conf.beta2))
    loss_criterion = nn.CrossEntropyLoss()
    loss_criterion_gan = nn.BCEWithLogitsLoss()

    steps = 0
    # writer = SummaryWriter()
    flag_D_strong = False
    for epoch in range(1,conf.epochs+1):
        print('%d epoch ...'%(epoch))
        g_loss = 0
        for i, batch_data in enumerate(train_loader):
            D.zero_grad()
            G.zero_grad()
            # print(batch_data[0].dtype)
            # print(type(batch_data[0]))
            batch_image = batch_data[0]
            batch_id_label = batch_data[1]-1
            batch_disguise_label = batch_data[2]
            batch_pro = batch_data[3]
            for j in range(conf.batch_size):
                if batch_disguise_label[j]==0:
                    batch_pro[j]=batch_image[j]
            batch_ones_label = t.ones(conf.batch_size)  
            batch_zeros_label = t.zeros(conf.batch_size)

            fixed_noise = t.FloatTensor(
                np.random.uniform(-1, 1, (conf.batch_size, conf.nz)))

            disguise_code_label = t.zeros(conf.batch_size).long()
            disguise_code = one_hot(disguise_code_label, conf.np)  # Condition 


            #cuda
            batch_image, batch_id_label, batch_disguise_label, batch_pro, batch_ones_label, batch_zeros_label = \
                batch_image.cuda(), batch_id_label.cuda(), batch_disguise_label.cuda(), batch_pro.cuda(), batch_ones_label.cuda(), batch_zeros_label.cuda()

            fixed_noise, disguise_code, disguise_code_label = \
                fixed_noise.cuda(), disguise_code.cuda(), disguise_code_label.cuda()


            generated = G(batch_image, fixed_noise)

            steps += 1

            if flag_D_strong:

                if i%5 == 0:
                    # Discriminator 
                  flag_D_strong, real_output,  syn_output = Learn_D(D, loss_criterion, loss_criterion_gan, optimizer_D, batch_image, batch_pro, generated, \
                                            batch_id_label, batch_disguise_label, batch_ones_label, batch_zeros_label, epoch, steps, conf.nd, conf)

                else:
                    # Generator
                    g_loss = Learn_G(D, loss_criterion, loss_criterion_gan, optimizer_G , batch_image, generated,\
                            batch_id_label, batch_disguise_label, batch_ones_label, disguise_code_label, epoch, steps, conf.nd, conf)
            else:

                if i%2==0:
                    # Discriminator 
                    flag_D_strong, real_output,  syn_output = Learn_D(D, loss_criterion, loss_criterion_gan, optimizer_D, batch_image, batch_pro, generated, \
                                            batch_id_label, batch_disguise_label, batch_ones_label, batch_zeros_label, epoch, steps, conf.nd, conf)

                else:
                    # Generator
                    g_loss = Learn_G(D, loss_criterion, loss_criterion_gan, optimizer_G , batch_image, generated, \
                            batch_id_label, batch_disguise_label, batch_ones_label, disguise_code_label, epoch, steps, conf.nd, conf)

            if i % 10 == 0:
                # x = vutils.make_grid(generated, normalize=True, scale_each=True)
                # writer.add_image('Image', x, i)
                generated = generated.cpu().data.numpy()/2+0.5
                batch_image = batch_image.cpu().data.numpy()/2+0.5
#                vis.images(generated,nrow=4,win='generated')
#                vis.images(batch_image,nrow=4,win='original')
                print('%d steps loss is  %f'%(steps,g_loss))
        if epoch%50 ==0:
            msg = 'Saving checkpoint :{}'.format(epoch)    #restore from epoch+1
            print(msg)
            G_state_list = G.state_dict()
            D_state_list = D.state_dict()
            t.save({
                'epoch':epoch,
                'g_net_list':G_state_list,
                'd_net_list' :D_state_list
            },
            os.path.join(conf.save_dir,'%04d.pth'% epoch))


def Learn_D(D_model, loss_criterion, loss_criterion_gan, optimizer_D, batch_image, batch_pro, generated, \
            batch_id_label, batch_disguise_label, batch_ones_label, batch_zeros_label, epoch, steps, Nd, args):

    real_output = D_model(batch_image)
    pro_output = D_model(batch_pro)
    syn_output = D_model(generated.detach()) # .detach() をすることで Generatorまでの逆伝播計算省略

    L_id    = loss_criterion(real_output[:, :Nd], batch_id_label)
    L_gan   = loss_criterion_gan(pro_output[:, Nd], batch_ones_label) + loss_criterion_gan(syn_output[:, Nd], batch_zeros_label)
    L_disguise  = loss_criterion(real_output[:, Nd+1:], batch_disguise_label)

    d_loss = L_gan + 5*L_id + 0.5*L_disguise

    d_loss.backward()
    optimizer_D.step()

    # Discriminator 
    flag_D_strong = Is_D_strong(real_output, syn_output, batch_id_label, batch_disguise_label, Nd)

    return flag_D_strong,  real_output,  syn_output



def Learn_G(D_model, loss_criterion, loss_criterion_gan, optimizer_G , batch_image, generated, \
            batch_id_label, batch_disguise_label, batch_ones_label, disguise_code_label, epoch, steps, Nd, args):

    syn_output=D_model(generated)

    L_id    = loss_criterion(syn_output[:, :Nd], batch_id_label)
    L_gan   = loss_criterion_gan(syn_output[:, Nd], batch_ones_label)
    L_disguise  = loss_criterion(syn_output[:, Nd+1:], disguise_code_label)

    Index=(batch_disguise_label==0).nonzero().squeeze()
    g_loss = L_gan + 5*L_id + 0.5*L_disguise + 0.1*(batch_image[Index] - generated[Index]).pow(2).sum()/args.batch_size
    g_loss.backward()
    optimizer_G.step()
    a = L_gan.cpu().data.item()
    return a


def Is_D_strong(real_output, syn_output, id_label_tensor, disguise_label_tensor, Nd, thresh=0.9):
    """
    # Discriminator 

    """
    _, id_real_ans = t.max(real_output[:, :Nd], 1)
    _, disguise_real_ans = t.max(real_output[:, Nd+1:], 1)
    _, id_syn_ans = t.max(syn_output[:, :Nd], 1)

    id_real_precision = (id_real_ans==id_label_tensor).type(t.FloatTensor).sum() / real_output.size()[0]
    disguise_real_precision = (disguise_real_ans==disguise_label_tensor).type(t.FloatTensor).sum() / real_output.size()[0]
    gan_real_precision = (real_output[:,Nd].sigmoid()>=0.5).type(t.FloatTensor).sum() / real_output.size()[0]
    gan_syn_precision = (syn_output[:,Nd].sigmoid()<0.5).type(t.FloatTensor).sum() / syn_output.size()[0]

    total_precision = (id_real_precision+disguise_real_precision+gan_real_precision+gan_syn_precision)/4

    # Variable(FloatTensor) 
    total_precision = total_precision.data.item()
    if total_precision>=thresh:
        flag_D_strong = True
    else:
        flag_D_strong = False

    return flag_D_strong


def generateDisguise(conf):
    vis = visdom.Visdom()
    G = Generator(conf.nz, 3).cuda()
    T=t.load('./saved_modelDisguise/H%3d.pth'%1600)    
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


            generated = G(batch_image, fixed_noise)
            
            
            steps += 1
            if i % 50 == 0:
                # x = vutils.make_grid(generated, normalize=True, scale_each=True)
                # writer.add_image('Image', x, i)
                generated = generated.cpu().data.numpy()/2+0.5
                batch_image = batch_image.cpu().data.numpy()/2+0.5
                batch_pro = batch_pro.cpu().data.numpy()/2+0.5
                
                vis.images(batch_image,nrow=4,win='original')
                vis.images(generated,nrow=4,win='generated')
                vis.images(batch_pro,nrow=4,win='prototype')
                                
if __name__=='__main__':
    conf = get_config()
    print(conf)
    if conf.TrainTag:
        trainDisguise(conf)
    else:
        generateDisguise(conf) 
    
