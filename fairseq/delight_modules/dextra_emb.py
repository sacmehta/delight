# ============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
# ============================================

import torch
from torch import nn
from fairseq.delight_modules.drop_layers import RecurrentDropout
from fairseq.delight_modules.dextra_unit import DExTraUnit
import math


class DExTraEmb(nn.Module):
    '''
        This class implements embeddings similar to DeFINE emebeddings introduced in
        https://arxiv.org/abs/1911.12385
    '''

    def __init__(self, args, map_layer, use_bias: bool = True):
        '''
        :param args: Argument list
        :param map_layer: Mapping layer (Adaptive or standard)
        :param use_bias: Use bias or not
        '''
        super(DExTraEmb, self).__init__()
        self.map_layer = map_layer
        self.input_features = args.delight_emb_map_dim
        self.embedding_dim = args.delight_emb_out_dim

        self.dextra_layer = DExTraUnit(
            in_features=self.input_features,
            in_proj_features=self.embedding_dim // 2,
            out_features=self.embedding_dim,
            width_multiplier=args.delight_emb_width_mult,
            dextra_depth=args.delight_emb_depth,
            dextra_dropout=args.delight_dropout,
            max_glt_groups=args.delight_emb_max_groups,
            act_type=args.act_type,
            norm_type=args.norm_type,
            use_bias=use_bias,
            is_iclr_version=args.define_iclr,
            glt_shuffle=args.glt_shuffle,
        )

        if args.adaptive_input:
            # added in Adaptive layer
            self.embed_scale = 1.0
        else:
            self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(self.input_features)

        self.drop_layer = RecurrentDropout(p=args.delight_emb_dropout, batch_first=True)

    def forward(self, x):
        '''
        B --> Batch size
        T --> Time steps
        E --> Embedding dimension
        :param x: Input of shape [B x T]
        :return: Output of shape [B x T x E]
        '''

        assert x.dim() == 2, 'Input should be [B x T]'

        # [B x T] --> [B x T x E]
        x = self.map_layer(x) * self.embed_scale
        # drop the embeddings
        x = self.drop_layer(x)
        # learn DeFINE embeddings
        x = self.dextra_layer(x)
        return x

    def __repr__(self):
        s = '{name}(in_features={input_features}, output_features={embedding_dim})'
        s += '\n \t {}'.format(self.map_layer)
        s += '\n \t {}'.format(self.dextra_layer)
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def compute_macs_params(self):
        emb_params = 0
        emb_macs = 0
        non_emb_macs = 0
        non_emb_params = 0

        from fairseq.modules.adaptive_input import AdaptiveInput
        if isinstance(self.map_layer, nn.Embedding):
            emb_params += self.map_layer.weight.numel()
            # LUT does not have any MACs
            emb_macs += 0
        elif isinstance(self.map_layer, AdaptiveInput):
            mac_params_adaptive = self.map_layer.compute_macs_params()

            emb_macs += mac_params_adaptive['embedding_macs']
            emb_params += mac_params_adaptive['embedding_params']

            non_emb_macs += mac_params_adaptive['proj_macs']
            non_emb_params += mac_params_adaptive['proj_params']

        macs_params_define = self.dextra_layer.compute_macs_params()

        non_emb_macs += macs_params_define['macs']
        non_emb_params += macs_params_define['params']

        # DeFINE embeddings can be cached, so non-emb MACS and PARAMS are zero
        # Uncomment below to see MACS and PARAMS during inference
        # non_emb_params = 0
        # non_emb_macs = 0

        return {
            'name': self.__class__.__name__,
            'emb_params': emb_params,
            'emb_macs': emb_macs,
            'non_emb_macs': non_emb_macs,
            'non_emb_params': non_emb_params
        }
