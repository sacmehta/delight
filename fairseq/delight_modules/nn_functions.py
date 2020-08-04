# ============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
# ============================================

import torch
from torch import nn
from torch.nn import functional as F
from fairseq.delight_modules.normalization_layers import get_norm_layer
from fairseq.delight_modules.activation_layers import get_activation_layer
from typing import Optional
from fairseq.delight_modules.print_utilities import *


class GroupLinear(nn.Module):
    '''
        This class implements the Grouped Linear Transform
        This is based on the Pyramidal recurrent unit paper:
            https://arxiv.org/abs/1808.09029
    '''

    def __init__(self, in_features: int, out_features: int, n_groups: int = 4, use_bias: bool = False,
                 use_shuffle: bool = False,
                 norm_type: Optional[str] = None, dropout: float = 0.0, act_type: Optional[str] = None):
        '''

        :param in_features: number of input features
        :param out_features: number of output features
        :param n_groups: number of groups in GLT
        :param use_bias: use bias or not
        :param use_shuffle: shuffle features between different groups
        :param norm_type: Normalization type (e.g. LayerNorm)
        :param dropout: Dropout value (default is 0.0)
        :param act_type: Activation type (e.g., Gelu or ReLU)
        '''
        super(GroupLinear, self).__init__()

        if in_features % n_groups != 0:
            err_msg = "Input dimensions ({}) must be divisible by n_groups ({})".format(in_features, n_groups)
            print_error_message(err_msg)
        if out_features % n_groups != 0:
            err_msg = "Output dimensions ({}) must be divisible by n_groups ({})".format(out_features, n_groups)
            print_error_message(err_msg)

        # warning_message = 'Please install custom cuda installation for faster training and inference'

        in_groups = in_features // n_groups
        out_groups = out_features // n_groups

        self.weights = nn.Parameter(torch.Tensor(n_groups, in_groups, out_groups))
        if use_bias:
            # add 1 in order to make it broadcastable across batch dimension
            self.bias = nn.Parameter(torch.Tensor(n_groups, 1, out_groups))
        else:
            self.bias = None

        if norm_type is not None:
            self.normalization_fn = get_norm_layer(name=norm_type, out_features=out_groups)
            self.norm_type = norm_type
        else:
            self.normalization_fn = None
            self.norm_type = None

        self.use_dropout = False
        self.drop_p = dropout
        if dropout > 0:
            self.drop_layer = nn.Dropout(p=dropout)
            self.use_dropout = True

        if act_type is not None:
            self.act_fn = get_activation_layer(name=act_type)
            self.act_type = act_type
        else:
            self.act_fn = None
            self.act_type = None

        self.n_groups = n_groups
        self.use_bias = use_bias
        self.shuffle = use_shuffle
        self.feature_shuffle = True if use_shuffle else False

        self.in_features = in_features
        self.out_features = out_features
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights.data)
        if self.use_bias:
            nn.init.constant_(self.bias.data, 0)

    def process_input_bmm(self, x):
        '''
        N --> Input dimension
        M --> Output dimension
        g --> groups
        G --> gates
        :param x: Input of dimension B x N
        :return: Output of dimension B x M
        '''
        bsz = x.size(0)
        # [B x N] --> [B x g  x N/g]
        x = x.contiguous().view(bsz, self.n_groups, -1)
        # [B x g x N/g] --> [g x B  x N/g]
        x = x.transpose(0, 1)  # transpose so that group is first

        # [g x B  x N/g] x [g x N/g x M/g] --> [g x B x M/g]
        x = torch.bmm(x, self.weights)  # multiply with Weights

        # add bias
        if self.use_bias:
            x = torch.add(x, self.bias)

        if self.feature_shuffle:
            # [g x B x M/g] --> [B x M/g x g]
            x = x.permute(1, 2, 0)
            # [B x M/g x g] --> [B x g x M/g]
            x = x.contiguous().view(bsz, self.n_groups, -1)
        else:
            # [g x B x M/g] --> [B x g x M/g]
            x = x.transpose(0, 1)  # transpose so that batch is first

        # feature map normalization
        if self.normalization_fn is not None:
            x = self.normalization_fn(x)

        # feature map activation (or thresholding)
        if self.act_fn is not None:
            x = self.act_fn(x)

        return x

    def forward(self, x):
        '''
        :param x: Input of shape [T x B x N] (should work with [B x T x N]
        :return:
        '''
        if x.dim() == 2:
            x = self.process_input_bmm(x)
        elif x.dim() == 3:
            T, B, N = x.size()
            x = x.contiguous().view(B * T, -1)
            x = self.process_input_bmm(x)
            x = x.contiguous().view(T, B, -1)
        else:
            raise NotImplementedError

        # dropout
        if self.use_dropout:
            x = self.drop_layer(x)
        return x

    def __repr__(self):
        s = '{name}(in_features={in_features}, out_features={out_features}, num_groups={n_groups}'
        if self.use_bias:
            s += ', bias={use_bias}'
        if self.shuffle:
            s += ', shuffle={shuffle}'

        if self.norm_type is not None:
            s += ', norm_type={norm_type}'
        if self.act_type is not None:
            s += ', act_type={act_type}'
        if self.drop_p > 0.0:
            s += ', drop_p={drop_p}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def compute_macs_params(self):
        '''
            # of operations in group linear transformation (GLT) are given as:
            Let N and M be dimensions of the input and the output tensor
            Both input and output are split into G groups, so that each input and output group has dimension of N/G and M/G
            Each input group of dimension N/G is mapped to each output group of dimension M/G using a matrix with dimensions [N/G x M/G].
            This mapping involves NM/G^2 additions and NM/G^2 multiplications.
            Since, there are G such groups, we will have total of NM/G addiations and NM/G multipplications.
            Or in simple words, total multiplication-additions (MACs) would be NM/G and FLOPs would be 2NM/G.

            Relationship with # of parameters:
            We have G matrices, each of dimension [N/G x M/G]. The number of parameters in each matrix is NM/G^2.
            Therefore, the total number of parameters in GLT is NM/G.

            MACs = parameters
        '''
        n_mul_wt = self.weights.numel()
        n_add_bias = self.bias.numel() if self.use_bias else 0
        macs = n_mul_wt + n_add_bias
        n_params = n_mul_wt + n_add_bias

        if self.normalization_fn is not None:
            n_params += sum([p.numel() for p in self.normalization_fn.parameters()])
            # MACS are zero for LayerNorm because they can be fused

        return {
            'name': self.__class__.__name__,
            'macs': macs,
            'params': n_params
        }


class Linear(nn.Module):
    '''
    This class implements the fully connected layer
    '''

    def __init__(self, in_features, out_features, use_bias=True, num_gates=1,
                 norm_type=None, dropout=0.0, act_type=None):
        '''
        :param in_features: number of input features
        :param out_features: number of output features
        :param use_bias: use bias or not
        :param num_gates: number of gates (useful if you want to use it within gating structures, like LSTMs)
        :param norm_type: Normalization type (e.g. LayerNorm)
        :param dropout: Dropout value (default is 0.0)
        :param act_type: Activation type (e.g., Gelu or ReLU)
        '''
        super(Linear, self).__init__()

        self.weights = torch.nn.Parameter(torch.Tensor(out_features * num_gates, in_features))
        if use_bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features * num_gates))
        else:
            self.bias = None

        if norm_type is not None:
            self.normalization_fn = get_norm_layer(name=norm_type, out_features=out_features * num_gates)
            self.norm_type = norm_type
        else:
            self.normalization_fn = None
            self.norm_type = None

        self.use_dropout = False
        self.drop_p = dropout
        if dropout > 0:
            self.drop_layer = nn.Dropout(p=dropout)
            self.use_dropout = True

        if act_type is not None:
            self.act_fn = get_activation_layer(name=act_type)
            self.act_type = act_type
        else:
            self.act_fn = None
            self.act_type = None

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.gates = num_gates
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights.data)
        if self.use_bias:
            nn.init.constant_(self.bias.data, 0)

    def forward(self, x):
        '''
        :param x: Input
        :return: Output
        '''
        x = F.linear(x, weight=self.weights, bias=self.bias)
        # feature map normalization
        if self.normalization_fn is not None:
            x = self.normalization_fn(x)

        # feature map activation (or thresholding)
        if self.act_fn is not None:
            x = self.act_fn(x)

        # recurrent dropout
        if self.use_dropout:
            x = self.drop_layer(x)

        return x

    def __repr__(self):
        s = '{name}(in_features={in_features}, out_features={out_features}'

        if self.use_bias:
            s += ', bias={use_bias}'

        if self.gates > 1:
            s += ', gates={gates}'

        if self.norm_type is not None:
            s += ', norm_type={norm_type}'
        if self.act_type is not None:
            s += ', act_type={act_type}'
        if self.drop_p > 0.0:
            s += ', drop_p={drop_p}'
        s += ')'

        return s.format(name=self.__class__.__name__, **self.__dict__)

    def compute_macs_params(self):
        '''
        # of operations in LT are given as:
            Let N and M be dimensions of the input and the output tensor
            Input dimension N is mapped to output of dimension M using a matrix with dimensions [N x M].
            This conversion will involve NM additions and NM multiplications.
            Or in simple words, total multiplication-additions (MACs) would be NM and FLOPs would be 2NM.

            Relationship with # of parameters:
            We have a matrix of dimension [N x M]. The number of parameters is NM.
            Therefore, the total number of parameters in LT is NM.

            MACs = parameters and FLOPs = 2 * parameters
        '''
        n_mul_wt = self.weights.numel()
        n_add_bias = self.bias.numel() if self.use_bias else 0
        macs = n_mul_wt + n_add_bias
        n_params = n_mul_wt + n_add_bias

        if self.normalization_fn is not None:
            n_params += sum([p.numel() for p in self.normalization_fn.parameters()])
            # MACS are zero for LayerNorm because they can be fused

        return {
            'name': self.__class__.__name__,
            'macs': macs,
            'params': n_params
        }


def get_weight_layer(name: str, in_features: int, out_features: int, groups: int = 4, use_bias: bool = True,
                     gates: int = 1, shuffle: bool = False,
                     norm_type: Optional[str] = None, dropout: float = 0.0, act_type: Optional[str] = None):
    # Group linear transform with groups=1 is the same as Linear Transformation
    if name == 'glt' and groups == 1:
        name = 'linear'

    if name == 'linear':
        layer = Linear(in_features=in_features, out_features=out_features, use_bias=use_bias, num_gates=gates,
                       norm_type=norm_type, dropout=dropout, act_type=act_type)
    elif name == 'glt':
        layer = GroupLinear(in_features=in_features, out_features=out_features, n_groups=groups,
                            use_bias=use_bias, use_shuffle=shuffle, norm_type=norm_type,
                            dropout=dropout, act_type=act_type)
    else:
        raise NotImplementedError
    return layer

def get_embedding_layer(num_embeddings, embedding_dim, padding_idx=None):
    emb = nn.Embedding(num_embeddings, embedding_dim, padding_idx)
    # initialize embedding layer
    nn.init.normal_(emb.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(emb.weight[padding_idx], 0)
    return emb



if __name__ == '__main__':
    bsz=10
    groups = 4
    n_in = 100
    n_out = 100

    for n_in in [32, 64, 128, 2048]:
        for n_out in [32, 64, 128, 2048]:
            for groups in [2, 4, 16, 32]:

                a = torch.Tensor(bsz, n_in).random_(1, 255).cuda()
                layer = GroupLinear(in_features=n_in, out_features=n_out, n_groups=groups, use_bias=False, use_shuffle=False,
                                    norm_type=None).cuda()
                nn.init.uniform_(layer.weights.data, 1, 200)
                layer.weights.data = torch.round(layer.weights.data)
                out = layer(a)
                #out_2 = layer(a, use_matmul=False)

                # [B x N] -->  [B x g x N/g]
                x = a.contiguous().view(bsz, groups, -1)
                # [B x g x N/g] --> [g x B  x N/g]
                x = x.transpose(0, 1)  # transpose so that group is first
                # [g x B  x N/g] x [g x N/g x GM/g] --> [g x B x GM/g]
                x = torch.bmm(x, layer.weights)  # multiply with Weights
                bmm_out = x.transpose(0, 1)

                diff = torch.sum(torch.abs(out - bmm_out))
                if diff != 0.0:
                    print(out)
                    print(bmm_out)
                    print('Failed for config: {}, {}, {}, {}'.format(n_in, n_out, groups, diff))
                    exit()
                else:
                    print('Success for config: {}, {}, {}, {}'.format(n_in, n_out, groups, diff))