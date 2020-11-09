import torch
from torch import nn, optim
class Discriminator(nn.Module):
    """
    multi-task CNN for identity classciation and variation detection

    ### init
    Nd : Number of identitiy to classify
    Np : Np =2. Has variation / Has no variation 

    """

    def __init__(self, Nd, Np, channel_num):
        super(Discriminator, self).__init__()
        convLayers = [
            nn.Conv2d(channel_num, 32, 3, 1, 1, bias=False), # Bxchx96x96 -> Bx32x96x96
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False), # Bx32x96x96 -> Bx64x96x96
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, 4, 2, 1, bias=False), # Bx64x96x96 -> Bx64x48x48
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 128, 3, 1, 1, bias=False), # Bx64x48x48 -> Bx128x48x48
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Conv2d(128, 128, 4, 2, 1, bias=False), #  Bx128x48x48 -> Bx128x24x24
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Conv2d(128, 256, 3, 1, 1, bias=False), #  Bx128x24x24 -> Bx256x24x24
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.Conv2d(256, 256, 4, 2, 1, bias=False), #  Bx256x24x24 -> Bx256x12x12
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.Conv2d(256, 256, 4, 2, 1, bias=False),  # Bx256x12x12 -> Bx256x6x6
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.Conv2d(256, 320, 3, 1, 1, bias=False), # Bx256x6x6 -> Bx320x6x6
            nn.ELU(),
            nn.AvgPool2d(6, stride=1), #  Bx320x6x6 -> Bx320x1x1
        ]

        self.convLayers = nn.Sequential(*convLayers)
        self.fc = nn.Linear(320, Nd+1+Np)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)

            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)

    def forward(self, input):
        x = self.convLayers(input)
        x = x.view(-1, 320)
        x = self.fc(x) # Bx320 -> B x (Nd+3)

        return x

class Crop(nn.Module):

    def __init__(self, crop_list):
        super(Crop, self).__init__()

        self.crop_list = crop_list

    def forward(self, x):
        B,C,H,W = x.size()
        x = x[:,:, self.crop_list[0] : H - self.crop_list[1] , self.crop_list[2] : W - self.crop_list[3]]

        return x


class Generator(nn.Module):
    """
    Encoder/Decoder conditional GAN conditioned with noise vector

    ### init
    Nz : Dimension of noise vector

    """

    def __init__(self, Nz, channel_num):
        super(Generator, self).__init__()
        self.features = []

        G_enc_convLayers = [
            nn.Conv2d(channel_num, 32, 3, 1, 1, bias=False), # Bxchx96x96 -> Bx32x96x96
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False), # Bx32x96x96 -> Bx64x96x96
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, 4, 2, 1, bias=False), # Bx64x96x96 -> Bx64x48x48
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 128, 3, 1, 1, bias=False), # Bx64x48x48 -> Bx128x48x48
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Conv2d(128, 128, 4, 2, 1, bias=False), #  Bx128x48x48 -> Bx128x24x24
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Conv2d(128, 256, 3, 1, 1, bias=False), #  Bx128x24x24 -> Bx256x24x24
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.Conv2d(256, 256, 4, 2, 1, bias=False), #  Bx256x24x24 -> Bx256x12x12
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.Conv2d(256, 256, 4, 2, 1, bias=False),  # Bx256x12x12 -> Bx256x6x6
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.Conv2d(256, 320, 3, 1, 1, bias=False), # Bx256x6x6 -> Bx320x6x6
            nn.ELU(),
            nn.AvgPool2d(6, stride=1), #  Bx320x6x6 -> Bx320x1x1
        ]
        self.G_enc_convLayers = nn.Sequential(*G_enc_convLayers)

        G_dec_convLayers = [
            nn.ConvTranspose2d(320,256, 3,1,1, bias=False), # Bx320x6x6 -> Bx256x6x6
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.ConvTranspose2d(256, 256, 4,2,1, bias=False), # Bx256x6x6 -> Bx256x12x12
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.ConvTranspose2d(256, 256, 4,2,1, bias=False), # Bx256x12x12 -> Bx256x24x24
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.ConvTranspose2d(256, 128,  3,1,1, bias=False), # Bx256x24x24 -> Bx128x24x24
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.ConvTranspose2d(128, 128, 4,2,1, bias=False), # Bx128x24x24 -> Bx128x48x48
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.ConvTranspose2d(128, 64, 3,1,1, bias=False), # Bx128x48x48 -> Bx64x48x48
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.ConvTranspose2d(64, 64, 4,2,1, bias=False), # Bx64x48x48 -> Bx64x96x96
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.ConvTranspose2d(64, 32, 3,1,1, bias=False), # Bx64x96x96 -> Bx32x96x96
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.ConvTranspose2d(32, channel_num, 3,1,1, bias=False), # Bx32x96x96 -> Bxchx96x96
            nn.Tanh(),
        ]

        self.G_dec_convLayers = nn.Sequential(*G_dec_convLayers)

        self.G_dec_fc = nn.Linear(320+Nz, 320*6*6)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)

            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)

            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)



    def forward(self, input, noise):

        x = self.G_enc_convLayers(input) # Bxchx96x96 -> Bx320x1x1

        x = x.view(-1,320)

        self.features = x

        x = torch.cat([x, noise], 1)  # Bx320 -> B x (320+Nz)

        x = self.G_dec_fc(x) # B x (320+Nz) -> B x (320x6x6)

        x = x.view(-1, 320, 6, 6) # B x (320x6x6) -> B x 320 x 6 x 6

        x = self.G_dec_convLayers(x) #  B x 320 x 6 x 6 -> Bxchx96x96

        return x