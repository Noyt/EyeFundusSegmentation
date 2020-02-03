import torch
from torch import nn
import torch.nn.functional as F
import utils.tools as tool
from networks.abstractnet import AbstractNet


class UNet(AbstractNet):
    """
    Implements the classic UNet, as described in the original Ronneberger 2015 paper.
    """
    def __init__(self,
                 config,
                 checkpoint='../trained_models/'
                 ):
        super().__init__(checkpoint=checkpoint)
        self.config = config
        "Descending path"
        self.downPath = nn.ModuleList()
        prev_channels = self.config.input_chan
        for i in range(self.config.depth):
            self.downPath.append(ConvBlock(config, in_channels=prev_channels, out_channels=2**(self.config.convLayerFactor + i)))
            prev_channels = 2**(self.config.convLayerFactor + i)

        "Lower U net path"
        self.lowPath = ConvBlock(config, in_channels=prev_channels, out_channels=2*prev_channels)
        prev_channels = 2*prev_channels

        "Ascending path"
        self.upPath = nn.ModuleList()
        "TODO check index"
        for i in reversed(range(self.config.depth)):
            self.upPath.append(UpConvBlock(config, in_channels=prev_channels, out_channels=2**(self.config.convLayerFactor + i)))
            prev_channels = 2**(self.config.convLayerFactor + i)

        "Final part"
        self.finalConv = nn.Conv2d(2**self.config.convLayerFactor, self.config.n_classes, kernel_size=3, padding=self.config.conv_pad)

    def forward(self, input):
        results = []
        "Going through down path, applying max pools, and saving intermediary results for skip connections"
        for down_block in self.downPath:
            input = down_block(input)
            results.append(input)
            input = F.max_pool2d(input, 2, stride=2)

        "Low part of the net"
        input = self.lowPath(input)

        "Going through up path, applying up-convs, and using skip connections"
        for i, up_block in enumerate(self.upPath):
            input = up_block(input, results[-1-i])

        return self.finalConv(input)


class ConvBlock(nn.Module):
    """
    Applies two successive convolutions. Does not perform max pooling or upsampling.
    """
    def __init__(self,
                 config,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 dilation=1,
                 groups=1,
                 padding_mode='zeros'):
        super(ConvBlock, self).__init__()
        self.config = config
        act = self.config.activation
        operations = []
        operations.append(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=self.config.conv_pad)
        )

        if act == 'selu':
            operations.append(nn.SELU())
        elif act == 'relu':
            operations.append(nn.ReLU(inplace=True))
        elif act == 'leakyrelu':
            operations.append(nn.LeakyReLU(negative_slope=0.01, inplace=True))
        elif act == 'elu':
            operations.append(nn.ELU())

        operations.append(
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=self.config.conv_pad)
        )

        if act == 'selu':
            operations.append(nn.SELU())
        elif act == 'relu':
            operations.append(nn.ReLU(inplace=True))
        elif act == 'leakyrelu':
            operations.append(nn.LeakyReLU(negative_slope=0.01, inplace=True))
        elif act == 'elu':
                operations.append(nn.ELU())

        self.block = nn.Sequential(*operations)

    def forward(self, input):
        return self.block(input)


class UpConvBlock(nn.Module):
    """
    Used during the ascending path when skip connections are required. Includes a ConvBlock and adds an additional
    upsampling operation.
    """
    def __init__(self,
                 config,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 upC_k_size=2,
                 ):
        super(UpConvBlock, self).__init__()
        self.config = config
        """
        if self.config.upsampling == 'transposed':
            "Up-conv using transposed convolution "
            self.upConv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=upC_k_size, padding=1, stride=2,  output_padding=1)
        elif self.config.upsampling == 'nearest':
            "Up conv replaced by simple upsampling"
            self.upConv = nn.Upsample(in_channels, in_channels, scale_factor=2, mode=self.config.upsampling)
        """
        self.upConv = Upsampling(in_channels, out_channels, upsampling=self.config.upsampling)
        self.convBlock = ConvBlock(config, in_channels, out_channels, kernel_size=3)

    def forward(self, input, skipInput):
        input = self.upConv(input)
        #cropped = tool.crop(skipInput, input.shape[2:])
        #output = torch.cat([input, cropped], 1)
        output = torch.cat([input, skipInput], 1)
        output = self.convBlock(output)
        return output


class Upsampling(nn.Module):
    def __init__(self, input_chan, output_chan, upsampling='nearest'):
        super(Upsampling, self).__init__()

        self.upsampling = upsampling

        if upsampling == 'transposed':
            self.op = nn.ConvTranspose2d(input_chan, output_chan, kernel_size=3, padding=1,
                                         stride=2,
                                         output_padding=1)
        else:
            self.op = nn.Conv2d(in_channels=input_chan,
                                out_channels=output_chan,
                                kernel_size=3,
                                padding=1, stride=1)

    def forward(self, input):

        if self.upsampling != 'transposed':
            input = nn.functional.interpolate(input, scale_factor=2, mode=self.upsampling)
        return self.op(input)