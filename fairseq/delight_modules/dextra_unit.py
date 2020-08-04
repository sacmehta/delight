# ============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
# ============================================

import torch
from torch import nn
from fairseq.delight_modules.nn_functions import get_weight_layer
from fairseq.delight_modules.activation_layers import get_activation_layer
import math
import numpy as np
from typing import Optional
from fairseq.delight_modules.print_utilities import *


class DExTraUnit(nn.Module):
    '''
        This class implements the DeFINE unit and the DeXTRA unit introduced in
        DeFINE unit: https://arxiv.org/abs/1911.12385
        DeXTRA Unit: https://arxiv.org/pdf/2008.00623.pdf
    '''

    def __init__(self,
                 in_features: int,
                 in_proj_features: int,
                 out_features: int,
                 width_multiplier: float = 2.0,
                 dextra_depth: int = 4,
                 dextra_dropout: float = 0.1,
                 max_glt_groups: int = 8,
                 act_type: str = 'gelu',
                 norm_type: str = 'ln',
                 use_bias: bool = True,
                 glt_shuffle: bool = False,
                 is_iclr_version: bool = False,
                 *args, **kwargs):
        '''
        :param in_features: Input features
        :param in_proj_features: Projected features for the first layer
        :param out_features: Output features
        :param width_multiplier: Width multiplier. Max. dimension in DExTra or DeFINE is width_multiplier * in_features
        :param dextra_depth: Number of GLT layers
        :param dextra_dropout: Dropout value between GLT layers
        :param max_glt_groups: Max groups in GLT
        :param act_type: Activation function
        :param norm_type: Normalization function
        :param use_bias: Bias or not
        :param glt_shuffle: Feature shuffling in GLT or not
        :param is_iclr_version: Using DeFINE or Dextra (default is Dextra)
        :param args: Unused args
        :param kwargs: Unused kwargs
        '''
        super(DExTraUnit, self).__init__()
        assert dextra_depth > 1, 'We need atleast 2 layers for DeFINE'
        assert in_features % 2 == 0, '# of Input features should be divisible by 2'
        assert in_features % max_glt_groups == 0, '# of Input features ({}) should be divisible by max groups ({})'.format(
            in_features, max_glt_groups)

        self.in_features = in_features
        self.in_proj_features = in_proj_features
        self.out_features = out_features
        self.width_multiplier = width_multiplier
        self.max_features = in_features * self.width_multiplier
        self.num_glt_layers = dextra_depth
        self.max_glt_groups = max_glt_groups
        self.dextra_dropout = dextra_dropout
        self.act_type = act_type
        self.norm_type = norm_type
        self.glt_shuffle = False if is_iclr_version else glt_shuffle  # no shuffling in ICLR version

        self.input_layer = get_weight_layer(name='linear',
                                            in_features=self.in_features,
                                            out_features=self.in_proj_features,
                                            use_bias=True,
                                            norm_type=self.norm_type,
                                            act_type=self.act_type,
                                            dropout=self.dextra_dropout
                                            )

        # get config for Group linear transformations
        if is_iclr_version:
            layer_config = self.define_config(in_features=self.in_proj_features,
                                              out_features=self.out_features,
                                              max_features=self.max_features,
                                              n_layers=self.num_glt_layers,
                                              max_groups=self.max_glt_groups)
        else:
            layer_config = self.dextra_config(in_features=self.in_proj_features,
                                              out_features=self.out_features,
                                              max_features=self.max_features,
                                              n_layers=self.num_glt_layers,
                                              max_groups=self.max_glt_groups
                                              )

        # setup expansion and reduction
        dextra_layers = nn.ModuleList()
        groups_next_layer = layer_config['groups'][1:] + [1]

        for idx, (n_in, n_out, g_l, g_l1) in enumerate(zip(layer_config['in'],
                                                           layer_config['out'],
                                                           layer_config['groups'],
                                                           groups_next_layer)):
            wt_layer = get_weight_layer(name='glt', in_features=n_in,
                                        out_features=n_out,
                                        groups=g_l,
                                        use_bias=use_bias,
                                        norm_type=self.norm_type,
                                        act_type=self.act_type,
                                        dropout=self.dextra_dropout,
                                        shuffle=self.glt_shuffle
                                        )

            dextra_layers.append(wt_layer)

        self.output_layer = get_weight_layer(name='linear',
                                             in_features=self.out_features + self.in_proj_features,
                                             out_features=self.out_features,
                                             use_bias=True,
                                             norm_type=norm_type,
                                             act_type=act_type,
                                             dropout=dextra_dropout
                                             )

        self.dextra_layers = dextra_layers
        self.groups_per_layer = layer_config['groups']

    def __repr__(self):
        s = '{name}(in_features={in_features}, out_features={out_features}, width_multiplier={width_multiplier}, ' \
            'normalization={norm_type}, activation={act_type}, dextra_dropout={dextra_dropout})'
        s += '\n  \t |---- {}'.format(self.input_layer)
        for layer_name in self.dextra_layers:
            s += '\n  \t |---- {}'.format(layer_name)
        s += '\n  \t |---- {}'.format(self.output_layer)
        return s.format(name=self.__class__.__name__, **self.__dict__)

    @staticmethod
    def dextra_config(in_features, out_features, max_features, n_layers, max_groups):

        mid_point = int(math.ceil(n_layers / 2.0))
        # decide number of groups per layer
        groups_per_layer = [min(2 ** (i + 1), max_groups) for i in range(mid_point)]

        # divide the space linearly between input_features and max_features
        output_sizes = np.linspace(in_features, max_features, mid_point, dtype=np.int).tolist()
        # invert lists to get the reduction groups and sizes
        inv_output_sizes = output_sizes[::-1]
        inv_group_list = groups_per_layer[::-1]
        if n_layers % 2 == 0:
            # even
            groups_per_layer = groups_per_layer + inv_group_list
            output_sizes = output_sizes + inv_output_sizes
        else:
            # for odd case,
            groups_per_layer = groups_per_layer + inv_group_list[1:]
            output_sizes = output_sizes + inv_output_sizes[1:]

        assert len(output_sizes) == len(groups_per_layer), '{} != {}'.format(len(output_sizes), len(groups_per_layer))
        output_sizes = output_sizes[:-1]

        # ensure that output and input sizes are divisible by group size
        input_sizes = [1] * len(groups_per_layer)
        input_sizes[0] = in_features
        for i in range(n_layers - 1):
            # output should be divisible by ith groups as well as i+1th group
            # Enforcing it to be divisble by 8 so that we can maximize tensor usage
            g_l = max(groups_per_layer[i + 1], groups_per_layer[i], 8)
            out_dim_l = int(math.ceil(output_sizes[i] / g_l)) * g_l
            inp_dim_l1 = out_dim_l + in_features

            if out_dim_l % 8 != 0:
                print_warning_message(
                    'To maximize tensor usage, output dimension {} should be divisible by 8'.format(out_dim_l))

            if inp_dim_l1 % 8 != 0:
                print_warning_message(
                    'To maximize tensor usage, input dimension {} should be divisible by 8'.format(inp_dim_l1))

            input_sizes[i + 1] = inp_dim_l1
            output_sizes[i] = out_dim_l

        # add dimensions corresponding to reduction step too
        output_sizes = output_sizes + [out_features]

        return {'in': input_sizes,
                'out': output_sizes,
                'groups': groups_per_layer
                }

    @staticmethod
    def define_config(in_features, out_features, max_features, n_layers, max_groups):
        # decide number of groups per layer
        groups_per_layer = []
        counter = 0
        for i in range(n_layers):
            g = 2 ** counter
            if g <= max_groups:
                counter += 1
            else:
                # reset
                g = 1  # set the current groups to 1
                counter = 1  # so that next group has 2 groups
            groups_per_layer.append(g)

        groups_per_layer = groups_per_layer[::-1]

        # divide the space linearly between input_features and max_features
        output_sizes = np.linspace(in_features, max_features, n_layers)
        output_sizes = output_sizes.astype(np.int).tolist()
        output_sizes = output_sizes[1:]

        # ensure that output and input sizes are divisible by group size
        input_sizes = [1] * len(groups_per_layer)
        input_sizes[0] = in_features
        for i in range(n_layers - 1):
            # output should be divisible by ith groups as well as i+1th group
            g_l = max(groups_per_layer[i + 1], groups_per_layer[i], 8)
            out_dim_l = int(math.ceil(output_sizes[i] / g_l)) * g_l
            inp_dim_l1 = out_dim_l + in_features

            if out_dim_l % 8 != 0:
                print_warning_message(
                    'To maximize tensor usage, output dimension {} should be divisible by 8'.format(out_dim_l))

            if inp_dim_l1 % 8 != 0:
                print_warning_message(
                    'To maximize tensor usage, input dimension {} should be divisible by 8'.format(inp_dim_l1))

            input_sizes[i + 1] = inp_dim_l1
            output_sizes[i] = out_dim_l

        # add dimensions corresponding to reduction step too
        output_sizes = output_sizes + [out_features]

        return {'in': input_sizes,
                'out': output_sizes,
                'groups': groups_per_layer
                }

    def forward_dextra(self, x):
        '''
        T -- > time steps
        B --> Batch size
        N, M --> Input, output features
        :param x: Input is [TxBxN] or [BxTxN]
        :return: output is [TxBxM] or [BxTxM]
        '''
        B = x.size(0)
        T = x.size(1)

        out = x

        for i, layer_i in enumerate(self.dextra_layers):
            # Transform Layer
            out = layer_i(out)

            g_next_layer = self.groups_per_layer[i + 1] if i < self.num_glt_layers - 1 else 1
            if g_next_layer == 1:
                # Linear layer is connected to everything so shuffle and split is useless for G=1
                out = torch.cat([x, out], dim=-1)
            else:
                # SPLIT and MIX LAYER
                # [B x T x M] --> [B x T x  G x M/G]
                x_g = x.contiguous().view(B, T, g_next_layer, -1)

                out = out.contiguous().view(B, T, g_next_layer, -1)

                # [B x T x G x M / G] || [B x T x G x N/G] --> [B x T x G x N+ M/G]
                out = torch.cat([x_g, out], dim=-1)

                # [B x T x G x N+ M/G] --> [B x T x N + M]
                out = out.contiguous().view(B, T, -1)

        out = self.output_layer(out)
        return out

    def forward(self, x):
        '''
        :param x: Input is [B x T x N]
        :return: Output is [B x T x M]
        '''

        # process input
        x = self.input_layer(x)
        n_dims = x.dim()

        if n_dims == 2:
            # [B x N] --> [B x 1 x N]
            x = x.unsqueeze(dim=1)  # add dummy T dimension
            # [B x 1 x N] --> [B x 1 x M]
            x = self.forward_dextra(x)
            # [B x 1 x M] --> [B x M]
            x = x.squeeze(dim=1)  # remove dummy T dimension
        elif n_dims == 3:
            x = self.forward_dextra(x)
        else:
            raise NotImplementedError
        return x

    def compute_macs_params(self):
        macs = 0
        n_params = 0

        macs_params_in = self.input_layer.compute_macs_params()
        macs += macs_params_in['macs']
        n_params += macs_params_in['params']

        macs_params_out = self.output_layer.compute_macs_params()
        macs += macs_params_out['macs']
        n_params += macs_params_out['params']

        for layer in self.dextra_layers:
            macs_params_define = layer.compute_macs_params()
            macs += macs_params_define['macs']
            n_params += macs_params_define['params']

        return {
            'name': self.__class__.__name__,
            'macs': macs,
            'params': n_params
        }


if __name__ == '__main__':
    B = 1
    T = 1
    G = 4
    a = torch.Tensor(B, T, 16).uniform_(1, 2)
    count = 0
    for i in range(16):
        if i % 2 == 0:
            count += 1
        a[:, :, i] = i + 1
    print(a)
    a = a.contiguous().view(B, T, G // 2, -1)
    a = a.permute(0, 1, 3, 2)
    # [B x T x M/G x  G] --> [B x T x G x  M/G]
    # Note that below is not a transpose operation.
    a = a.contiguous().view(B, T, G, -1)
    print(a)
    G = 8
    a = a.contiguous().view(B, T, G // 2, -1)
    a = a.permute(0, 1, 3, 2)
    # [B x T x M/G x  G] --> [B x T x G x  M/G]
    # Note that below is not a transpose operation.
    a = a.contiguous().view(B, T, G, -1)
    print(a)
