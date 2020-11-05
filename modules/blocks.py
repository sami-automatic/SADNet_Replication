import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np

from dcn.deform_conv import ModulatedDeformConvPack2 as DCN

class ResBlock(nn.Module):

    def __init__(self, in_channels=32, out_channels=32):
        super().__init__()

        self.conv1 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        #self.initialize_weights()

    def forward(self, x):
        conv1 = self.lrelu(self.conv1(x))
        conv2 = self.conv2(conv1)
        out = x + conv2
        out = self.lrelu(out)
        return out


class RSABlock(nn.Module):

    def __init__(self, in_channels=32, out_channels=32, offset_channel=32):
        super().__init__()

        self.dcnpack = DCN(out_channels, out_channels, 3, stride=1, padding=1, dilation=1, deformable_groups=8,
                            extra_offset_mask=True, offset_in_channel=offset_channel)
        self.conv1 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        #self.initialize_weights()

    def forward(self, x, offset):
        fea = self.lrelu(self.dcnpack([x, offset]))
        out = self.conv1(fea) + x
        return out


class OffsetBlock(nn.Module):

    def __init__(self, in_channels=32, offset_channel=32, last_offset=False):
        super().__init__()
        self.offset_conv1 = nn.Conv2d(in_channels, offset_channel, 3, 1, 1)  # concat for diff
        if last_offset:
            self.offset_conv2 = nn.Conv2d(offset_channel*2, offset_channel, 3, 1, 1)  # concat for offset
        self.offset_conv3 = nn.Conv2d(offset_channel, offset_channel, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        #self.initialize_weights()

    def forward(self, x, last_offset=None):
        offset = self.lrelu(self.offset_conv1(x))
        if last_offset is not None:
            last_offset = F.interpolate(last_offset, scale_factor=2, mode='bilinear', align_corners=False)
            offset = self.lrelu(self.offset_conv2(torch.cat([offset, last_offset * 2], dim=1)))
        offset = self.lrelu(self.offset_conv3(offset))
        return offset


class ContextBlock(nn.Module):
    def __init__(self, in_channels=32, out_channels=32, square=False):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels, out_channels, 1, 1)
        if square:
            self.conv1 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, 1)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 2, 2)
            self.conv3 = nn.Conv2d(out_channels, out_channels, 3, 1, 4, 4)
            self.conv4 = nn.Conv2d(out_channels, out_channels, 3, 1, 8, 8)
        else:
            self.conv1 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, 1)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 2, 2)
            self.conv3 = nn.Conv2d(out_channels, out_channels, 3, 1, 3, 3)
            self.conv4 = nn.Conv2d(out_channels, out_channels, 3, 1, 4, 4)
        self.fusion = nn.Conv2d(4*out_channels, in_channels, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        #self.initialize_weights()

    def forward(self, x):
        x_reduce = self.conv0(x)
        conv1 = self.lrelu(self.conv1(x_reduce))
        conv2 = self.lrelu(self.conv2(x_reduce))
        conv3 = self.lrelu(self.conv3(x_reduce))
        conv4 = self.lrelu(self.conv4(x_reduce))
        out = torch.cat([conv1, conv2, conv3, conv4], 1)
        out = self.fusion(out) + x
        return out


