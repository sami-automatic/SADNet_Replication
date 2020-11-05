import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from dcn.deform_conv import ModulatedDeformConvPack2 as DCN
from modules.blocks import ResBlock, RSABlock, OffsetBlock, ContextBlock


class SADNET(nn.Module):

    def __init__(self, n_channel=32, offset_n_channel=32):
        super(SADNET).__init__()

        self.conv1 = nn.Conv2d(3, n_channel, kernel_size=1, stride=1)
        self.rb1 = ResBlock(n_channel, n_channel)
        
        self.down2 = nn.Conv2d(n_channel, n_channel*2, kernel_size=2, stride=2)
        self.rb2 = ResBlock (n_channel*2, n_channel*2)

        self.down3 = nn.Conv2d(n_channel*2, n_channel*4, kernel_size=2, stride=2)
        self.rb3 = ResBlock(n_channel*4, n_channel*4)  

        self.down4 = nn.Conv2d(n_channel*4, n_channel*8, kernel_size=2, stride=2)
        self.rb4 = ResBlock(n_channel*8, n_channel*8)

        self.context = ContextBlock(n_channel*8, n_channel*2, square=False)
        self.offset = OffsetBlock(n_channel*8, offset_channel=offset_n_channel, last_offset=False)
        self.rsab = RSABlock(n_channel*8, n_channel*8, offset_channel=offset_n_channel)

        self.up1_1 = nn.ConvTranspose2d(n_channel*8, n_channel*4, kernel_size=2, stride=2)
        self.up1_2 = nn.Conv2d(n_channel*8, n_channel*4, kernel_size=1, stride=1)    
        self.offset1 = OffsetBlock(n_channel*4, offset_channel=offset_n_channel, last_offset=True)
        self.rsab1 =  RSABlock(n_channel*4, n_channel*4, offset_channel=offset_n_channel)
    
        self.up2_1 = nn.ConvTranspose2d(n_channel*4, n_channel*2, kernel_size=2, stride=2)
        self.up2_2 = nn.Conv2d(n_channel*4, n_channel*2, kernel_size=1, stride=1)    
        self.offset2 = OffsetBlock(n_channel*2, offset_channel=offset_n_channel, last_offset=True)
        self.rsab2 =  RSABlock(n_channel*2, n_channel*2, offset_channel=offset_n_channel)

        self.up3_1 = nn.ConvTranspose2d(n_channel*2, n_channel, kernel_size=2, stride=2)
        self.up3_2 = nn.Conv2d(n_channel*2, n_channel, kernel_size=1, stride=1)    
        self.offset3 = OffsetBlock(n_channel, offset_channel=offset_n_channel, last_offset=True)
        self.rsab3 =  RSABlock(n_channel, n_channel, offset_channel=offset_n_channel)
    
        self.out = nn.Conv2d(n_channel, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        conv1 = F.leaky_relu(self.conv1(x),  negative_slope=0.2, inplace=True)
        conv1 = self.rb1(conv1)

        pool1= F.leaky_relu(self.down2(conv1), negative_slope=0.2, inplace=True)
        conv2 = self.rb2(pool1)

        pool2 = F.leaky_relu(self.down3(conv2), negative_slope=0.2, inplace=True)
        conv3 = self.rb3(pool2)

        pool3 = F.leaky_relu(self.down4(conv3), negative_slope=0.2, inplace=True)
        conv4 = self.rb4(pool3)
        conv4 = self.context(conv4)

        L4_offset = self.offset(conv4, None)
        conv4 = self.rsab(conv4, L4_offset)

        dconv1 = self.up1_1(conv4)
        concat1 = torch.cat([dconv1, conv3],1)
        concat1 = F.leaky_relu(self.up1_2(concat1), negative_slope=0.2, inplace=True)
        L3_offset = self.offset1(concat1, L4_offset)
        decoded1 = self.rsab1(concat1, L3_offset)

        dconv2 = self.up2_1(decoded1)
        concat2 = torch.cat([dconv2,conv2],1)
        concat2 = F.leaky_relu(self.up2_2(concat2),  negative_slope=0.2, inplace=True)
        L2_offset = self.offset2(concat2, L3_offset)
        decoded2 = self.rsab2(concat2, L2_offset)

        dconv3 = self.up3_1(decoded2)
        concat3 = torch.cat([dconv3, conv1],1)
        concat3 = F.leaky_relu(self.up3_2(concat3),  negative_slope=0.2, inplace=True)
        L1_offset = self.offset3(L2_offset, concat3)
        decoded3 = self.rsab3(concat3, L1_offset)

        out = self.out(decoded3) + x

        return out


if __name__ == '__main__':
    net = SADNET()

    img = torch.rand(1, 300, 300, 3)#patch base ? *16

    output = net(img)
    print(output.shape)
