# ============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
# ============================================

import torch
from torch import nn
from fairseq.delight_modules.print_utilities import *


class BatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(BatchNorm, self).__init__()
        self.layer = nn.BatchNorm1d(num_features=num_features, eps=eps, affine=affine)

    def forward(self, x):
        if x.dim() == 3:
            bsz, seq_len, feature_size = x.size()
            out = self.layer(x.view(-1, feature_size))
            return out.contiguous().view(bsz, seq_len, -1)
        else:
            return self.layer(x)

norm_layer_list = [
    'gn', 'bn', 'ln'
]

def get_norm_layer(name, out_features, num_groups=1, eps=1e-5, affine=True):
    if name == 'gn' and num_groups == 1:
        name = 'bn'

    if name == 'bn':
        return BatchNorm(num_features=out_features, eps=eps, affine=affine)
    elif name == 'ln':
        try:
            from apex.normalization import FusedLayerNorm
            return FusedLayerNorm(out_features, eps, affine)
        except:
            return nn.LayerNorm(out_features, eps=eps, elementwise_affine=affine)
    elif name == 'gn':
        return nn.GroupNorm(num_groups=num_groups, num_channels=out_features, eps=eps, affine=affine)
    else:
        print_error_message('Supported normalization functions: {}'.format(norm_layer_list))
        return None