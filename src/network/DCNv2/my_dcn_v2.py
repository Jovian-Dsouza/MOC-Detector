from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import math
from torch import nn
from torch.nn.modules.utils import _pair

class DCN(nn.Module):
    ## Convlotion wrapper function
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding,
                 dilation=1, deformable_groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                                stride
                                ,padding,
                                 dilation)

    def register_parameter(self):
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

    def forward(self, input):
        return self.conv(input)

    
    

