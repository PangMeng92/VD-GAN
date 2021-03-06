# VD-GAN

VD-GAN: A Unified Framework for Joint Prototype and Representation Learning from Contaminated Single Sample per Person

@article{pang2021vd,

  title={VD-GAN: A Unified Framework for Joint Prototype and Representation Learning From Contaminated Single Sample per Person},
  
  author={Pang, Meng and Wang, Binghui and Cheung, Yiu-ming and Chen, Yiran and Wen, Bihan},
  
  journal={IEEE Transactions on Information Forensics and Security},
  
  volume={16},
  
  pages={2246--2259},
  
  year={2021},
  
  publisher={IEEE}
  
}


This work has been published in IEEE Transactions on Information Forensics and Security (TIFs). In this package, we implement our VD-GAN using Pytorch, and train/test the VD-GAN model on CAS-PEAL (disguise) dataset. 

The PEAL dataset can be downloaded in the link (https://drive.google.com/file/d/1OqAA81yUXbyRIh0c8EJFAnOFbmvBGNMH/view?usp=sharing).

The trained VD-GAN model can be downloaded in the link (https://drive.google.com/file/d/1IliSX7Ma3D2F47P27eyz8Nlg0k_Q0I5u/view?usp=sharing).

----------------------------------------------------------------------------
Train VD-GAN model:

1. Open configDisguise.py 
set con.batch_size =16;
set conf.epochs = 2000;
set conf.file='./dataset/LoadPEAL200.txt';
conf.np = 2;
conf.nz = 50;
set conf.nd=200;
set conf.TrainTag = True;

2. Open readerDisguise.py
set shuffle=True in def get_batch

3. Run TrainPV_disguise.py
The trained model will be stored in saved_modelDisguise


----------------------------------------------------------------------------
Test VD-GAN model:

1. Open configDisguise.py 
set con.batch_size =16;
set conf.epochs = 2000;
set conf.file='./dataset/LoadPEAL100.txt';
conf.np = 2;
conf.nz = 50;
set conf.nd=100;
set conf.TrainTag = False;

2. Run TrainPV_disguise.py
Choose a trained model (e.g., H1600), and load it in def generateDisguise


----------------------------------------------------------------------------
Generate prototype images:

1. Open configDisguise.py 
set con.batch_size =1;
set conf.epochs = 1;
set conf.file='./dataset/LoadPEAL5.txt';
conf.np = 2;
conf.nz = 50;
set conf.TrainTag = False;

2. Open readerDisguise.py
set shuffle=False in def get_batch

3. Run TrainPV_disguise.py
Choose a trained model (e.g., H1600), and load it in def generateDisguise


PEAL_ori_test: input face images 

PEAL_gen_test: generated prototype images 

PEAL_pro_test: real prototype images 

The software is free for academic use, and shall not be used, rewritten, or adapted as the basis of a commercial product without first obtaining permission from the authors. The authors make no representations about the suitability of this software for any purpose. It is provided "as is" without express or implied warranty.






