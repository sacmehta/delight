# ============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
# ============================================

import torch
import math
from torch import nn
from fairseq.delight_modules.print_utilities import *


class GELU(torch.nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return torch.nn.functional.gelu(x)

class Swish(torch.nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)

activation_list = [
    'relu', 'leaky', 'selu', 'elu', 'celu', 'prelu', 'sigmoid', 'tanh', 'gelu', 'swish'
]

def get_activation_layer(name):
    if name == 'relu':
        return nn.ReLU(inplace=False)
    elif name == 'leaky':
        return nn.LeakyReLU(negative_slope=0.1, inplace=False)
    elif name == 'selu':
        return nn.SELU(inplace=True)
    elif name == 'elu':
        return nn.ELU(inplace=True)
    elif name == 'celu':
        return nn.CELU(inplace=True)
    elif name == 'prelu':
        return nn.PReLU()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'gelu':
        return GELU()
    elif name =='swish':
        return Swish()
    else:
        print_error_message('Supported activation functions: {}'.format(activation_list))
        return None
