# ============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
# ============================================

# Adapted from Transformer model in Fairseq

import math
from typing import Any, Dict, List, NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import options, utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    AdaptiveSoftmax,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    DeLighTTransformerEncoderLayer,
    DeLighTTransformerDecoderLayer,
)
from torch import Tensor
from fairseq.delight_modules.dextra_emb import DExTraEmb
from fairseq.delight_modules.normalization_layers import get_norm_layer
from fairseq.delight_modules.nn_functions import get_weight_layer, get_embedding_layer
from fairseq.delight_modules import (
    DEFAULT_WIDTH_MULTIPLIER,
    DEFAULT_MIN_DEXTRA_LAYERS,
    MIN_ELEMENTS_PER_GROUP,
    DEFAULT_FFN_RED_FACTOR,
    DEFAULT_DROPOUT,
    DEFAULT_MAX_DEXTRA_LAYERS
)
from fairseq.delight_modules.drop_layers import RecurrentDropout
import numpy as np
from fairseq.delight_modules.math_utils import bound_function
from fairseq.distributed_utils import is_master

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


@register_model("delight_transformer")
class DeLighTTransformerModel(FairseqEncoderDecoderModel):
    @classmethod
    def hub_models(cls):
        # fmt: off

        return None
        # fmt: on

    def __init__(self, args, encoder, decoder):
        super().__init__(encoder, decoder)
        self.args = args
        self.supports_align_args = True

    @staticmethod
    def add_args(parser):

        # MAP LAYER IN DEFINE
        parser.add_argument('--adaptive-input', action='store_true',
                            help='Use Adaptive input or standard embedding for mapping function in DeFINE')

        # ADAPTIVE INPUT AND OUTPUT SETTINGS
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        parser.add_argument('--adaptive-softmax-factor', type=float, metavar='N',
                            help='adaptive input factor')
        parser.add_argument('--tie-adaptive-weights', action='store_true',
                            help='if set, ties the weights of adaptive softmax and adaptive input')
        parser.add_argument('--tie-adaptive-proj', action='store_true',
                            help='if set, ties the projection weights of adaptive softmax and adaptive input')

        # DeLighT EMBEDDING
        parser.add_argument('--delight-emb-map-dim', type=int, help='Mapping dimension in look-up table')
        parser.add_argument('--delight-emb-out-dim', type=int, help='Output dimension of DeLighT embedding layer')
        parser.add_argument('--delight-emb-width-mult', type=int,
                            help='Expand define Embeddings by this factor')
        parser.add_argument('--delight-emb-max-groups', type=int,
                            help='Max. number of groups in DeLighT embedding layers')
        parser.add_argument('--delight-emb-dropout', type=float, help='Dropout in DeLighT embedding layers')
        parser.add_argument('--delight-emb-depth', type=int, help='Depth of DeLighT unit in embedding layer')

        # DeLighT ENCODER
        parser.add_argument('--delight-enc-scaling', type=str, choices=['block', 'uniform'],
                            help='Block-wise scaling or uniform')

        parser.add_argument('--delight-enc-layers', type=int, help='Number of DeLighT encoder layers')
        parser.add_argument('--delight-enc-min-depth', type=int, help='Min. number of DeLighT layers in encoder')
        parser.add_argument('--delight-enc-max-depth', type=int, help='Max. number of DeLighT layers in encoder')
        parser.add_argument('--delight-enc-width-mult', type=float, help='Encoder width multiplier')
        parser.add_argument('--delight-enc-ffn-red', type=int, help='Reduction factor in light-weight FFN')
        parser.add_argument('--delight-enc-max-groups', type=int, help='Max. groups in GLT')

        # DeLighT DECODER
        parser.add_argument('--delight-dec-scaling', type=str, choices=['block', 'uniform'],
                            help='Block-wise scaling or uniform')

        parser.add_argument('--delight-dec-layers', type=int, help='Number of DeLighT decoder layers')
        parser.add_argument('--delight-dec-min-depth', type=int, help='Min. number of DeLighT layers in decoder')
        parser.add_argument('--delight-dec-max-depth', type=int, help='Max. number of DeLighT layers in decoder')
        parser.add_argument('--delight-dec-width-mult', type=float, help='Decoder width multiplier')
        parser.add_argument('--delight-dec-ffn-red', type=int, help='Reduction factor in light-weight FFN')
        parser.add_argument('--delight-dec-max-groups', type=int, help='Max. groups in GLT')

        # DeLight COMMONS
        parser.add_argument('--no-glt-shuffle', action='store_true', help='Disable shuffling in GLT transformation')
        parser.add_argument('--define-iclr', action='store_true', help='DeFINE unit as in ICLR paper')
        parser.add_argument('--norm-type', type=str, help='Normalization layer')
        parser.add_argument('--act-type', type=str, help='Activation function')
        parser.add_argument('--delight-dropout', type=float, help='Dropout value for Delight layers')
        parser.add_argument('--ffn-dropout', type=float, help='Dropout after FFN')

        # Print  stats
        parser.add_argument('--print-stats', action='store_true', help='Print MACs')
        parser.add_argument('--src-len-ps', type=int, help='Source length for printing stats')
        parser.add_argument('--tgt-len-ps', type=int, help='Target length for printing stats')

        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--pe-dropout', type=float, metavar='D',
                            help='dropout probability for positional encodings')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')


        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')

        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')

        parser.add_argument('--no-scale-embedding', action='store_true',
                            help='if True, dont scale embeddings')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(args, dictionary):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            if args.adaptive_input:
                raise ValueError('Adaptive Input is not yet supported for NMT')
            else:
                map_layer = get_embedding_layer(num_embeddings=num_embeddings,
                                                embedding_dim=args.delight_emb_map_dim,
                                                padding_idx=padding_idx)

            emb = DExTraEmb(args, map_layer=map_layer)

            return emb

        encoder_embed_tokens = build_embedding(args, src_dict)
        decoder_embed_tokens = build_embedding(args, tgt_dict)
        if args.share_all_embeddings:
            if args.adaptive_input:
                raise ValueError('Adaptive Input is not yet supported for NMT')
            else:
                decoder_embed_tokens.map_layer.weight = encoder_embed_tokens.map_layer.weight

        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)

        # print macs and params layer-wise
        if args.print_stats and is_master(args):
            cls.comptue_stats(args, encoder, decoder)
        
        return cls(args, encoder, decoder)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        encoder = DeLighTTransformerEncoder(args, src_dict, embed_tokens)
        return encoder

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = DeLighTTransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )

        return decoder

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
            self,
            src_tokens,
            src_lengths,
            prev_output_tokens,
            cls_input: Optional[Tensor] = None,
            return_all_hiddens: bool = True,
            features_only: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            cls_input=cls_input,
            return_all_hiddens=return_all_hiddens,
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out

    @classmethod
    def comptue_stats(cls, args, encoder, decoder):

        target_length = args.tgt_len_ps
        source_length = args.src_len_ps

        format_str = "{:<20} | \t {:<10} | \t {:<10}"
        print('=' * 15 * source_length)
        print('{:<90} {:<20}'.format('', cls.__name__))
        print('=' * 15 * source_length)
        print(format_str.format('Layer', 'Params', 'MACs'))
        print('-' * 15 * source_length)
        overall_macs = 0.0
        overall_params = 0.0
        round_places = 2

        # encoder
        import csv
        with open('{}/encoder_stats_t_{}_s_{}.csv'.format(args.save_dir, target_length, source_length),
                  mode='w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            for enc_idx, (k, v) in enumerate(encoder.compute_macs_params(src_len=source_length).items()):
                macs = v['macs'] + v['emb_macs']
                params = v['params'] + v['emb_params']

                overall_macs += macs
                overall_params += params

                macs = round(float(macs) / 1e6, round_places)
                params = round(float(params) / 1e6, round_places)

                print(format_str.format(k, params, macs))

                if enc_idx == 0:
                    key_list = list(v.keys())
                    csv_writer.writerow(['Layer'] + key_list)

                value_list = list(v.values())
                value_list = [k] + value_list
                csv_writer.writerow(value_list)

        # print('-' * 60)
        # decoder

        dec_string = {}
        with open('{}/decoder_stats_t_{}_s_{}.csv'.format(args.save_dir, target_length, source_length),
                  mode='w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            for time_step in range(1, target_length + 1):
                for dec_idx, (k, v) in enumerate(
                        decoder.compute_macs_params(src_len=source_length, tgt_len=time_step).items()):

                    if args.share_all_embeddings and k == 'Dec_LUT':  # Look-up Table is shared
                        v['emb_params'] = 0

                    macs = v['macs'] + v['emb_macs']
                    params = v['params'] + v['emb_params']

                    overall_macs += macs
                    if time_step == 1:
                        overall_params += params

                    macs = round(float(macs) / 1e6, round_places)
                    params = round(float(params) / 1e6, round_places)

                    if k not in dec_string:
                        dec_string[k] = [[time_step, params, macs]]
                    else:
                        dec_string[k].append([time_step, params, macs])

                    if dec_idx == 0:
                        key_list = list(v.keys())
                        csv_writer.writerow(['Time'] + ['Layer'] + key_list)

                    value_list = list(v.values())
                    value_list = [time_step] + [k] + value_list
                    csv_writer.writerow(value_list)

        format_str_dec1 = '{:<20} | \t '.format("Layer")
        dotted_line = '-' * 20
        for t in range(target_length + 1):
            if t == 0:
                format_str_dec1 += '{:<10} | \t '.format("Params")
            else:
                format_str_dec1 += '{:<10} '.format("t_{}".format(t))
            dotted_line += '-' * 10
        dotted_line += '-' * 10
        format_str_dec1 += '| \t {:<10} '.format("Overall MAC")
        dotted_line += '-' * 10
        # print(format_str_dec)
        print(dotted_line)
        print(format_str_dec1)
        print(dotted_line)

        for layer_name, v in dec_string.items():
            time_step_str = '{:<20} | \t '.format(layer_name)
            macs = 0
            for idx, (t, p, m) in enumerate(v):
                # print(t)
                if idx == 0:
                    time_step_str += '{:<10} | \t '.format(p)
                    time_step_str += '{:<10} '.format(m)
                else:
                    time_step_str += '{:<10} '.format(m)
                macs += m
            time_step_str += '| \t {:<10} '.format(round(macs, round_places))
            print(time_step_str)
        overall_macs = round(float(overall_macs) / 1e6, round_places)
        overall_params = round(float(overall_params) / 1e6, round_places)
        print('-' * 15 * target_length)

        print('Total MACS for {} decoder timesteps: {} M'.format(target_length, overall_macs))
        print('Total parameters: {} M'.format(overall_params))
        print('=' * 15 * target_length)

        with open('{}/overall_stats_t_{}_s_{}.csv'.format(args.save_dir, target_length, source_length),
                  mode='w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(['Time steps', target_length])
            csv_writer.writerow(['Total MACs (in million)', overall_macs])
            csv_writer.writerow(['Total parameters (in million)', overall_params])


EncoderOut = NamedTuple(
    "EncoderOut",
    [
        ("encoder_out", Tensor),  # T x B x C
        ("encoder_padding_mask", Tensor)  # B x T
    ],
)


class DeLighTTransformerEncoder(FairseqEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))

        self.dropout = args.dropout

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.map_layer.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens

        self.embed_positions = (
            PositionalEmbedding(
                args.max_source_positions,
                embed_dim,
                self.padding_idx,
                learned=args.encoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

        self.positional_dropout = RecurrentDropout(p=args.pe_dropout, batch_first=True) if not args.no_token_positional_embeddings else None

        self.layer_wise_attention = getattr(args, "layer_wise_attention", False)

        self.layers = nn.ModuleList([])

        if args.delight_enc_scaling == 'block' and (args.delight_enc_min_depth == args.delight_enc_max_depth):
            args.delight_enc_scaling = 'uniform'


        if args.delight_enc_scaling == 'uniform':
            assert args.delight_enc_min_depth == args.delight_enc_max_depth
            self.layers.extend(
                [DeLighTTransformerEncoderLayer(args=args,
                                                embed_dim=embed_dim,
                                                width_multiplier=args.delight_enc_width_mult,
                                                dextra_depth=args.delight_enc_min_depth)
                 for _ in range(args.define_enc_layers)]
            )
        else:
            assert args.delight_enc_min_depth < args.delight_enc_max_depth

            dextra_depths = np.linspace(start=args.delight_enc_min_depth,
                                         stop=args.delight_enc_max_depth,
                                         num=args.delight_enc_layers,
                                         dtype=np.int)

            depth_ratio = (args.delight_enc_max_depth * 1.0) / args.delight_enc_min_depth

            width_multipliers = np.linspace(start=args.delight_enc_width_mult,
                                      stop=args.delight_enc_width_mult + (depth_ratio - 1.0), # subtraction by 1 for max==min case
                                      num=args.delight_enc_layers,
                                      dtype=np.float
                                      )

            self.layers.extend(
                [DeLighTTransformerEncoderLayer(args=args,
                                                embed_dim=embed_dim,
                                                width_multiplier=round(width_multipliers[idx], 3),
                                                dextra_depth=layer_i)
                 for idx, layer_i in enumerate(dextra_depths)
                 ]
            )

        self.num_layers = len(self.layers)

        if args.encoder_normalize_before:
            self.layer_norm = get_norm_layer(name=args.norm_type, out_features=embed_dim)
        else:
            self.layer_norm = None

    def forward( self, src_tokens, src_lengths, *args, **kwargs ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            namedtuple:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """

        x = self.embed_tokens(src_tokens)
        if self.embed_positions is not None:
            x += self.embed_positions(src_tokens)
            x = self.positional_dropout(x)
        else:
            x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask  # B x T
        )


    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out.encoder_out is not None:
            encoder_out = encoder_out._replace(
                encoder_out=encoder_out.encoder_out.index_select(1, new_order)
            )
        if encoder_out.encoder_padding_mask is not None:
            encoder_out = encoder_out._replace(
                encoder_padding_mask=encoder_out.encoder_padding_mask.index_select(
                    0, new_order
                )
            )
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if (
                not hasattr(self, "_future_mask")
                or self._future_mask is None
                or self._future_mask.device != tensor.device
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(tensor.new(dim, dim)), 1
            )
            if self._future_mask.size(0) < dim:
                self._future_mask = torch.triu(
                    utils.fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1
                )
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                print("deleting {0}".format(weights_key))
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)
        for i in range(self.num_layers):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(
                state_dict, "{}.layers.{}".format(name, i)
            )

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict

    def compute_macs_params(self, src_len=1):
        encoder_mac_params = dict()

        emb_stats = self.embed_tokens.compute_macs_params()

        encoder_mac_params['Enc_LUT'] = {
            'macs': emb_stats['non_emb_macs'] * src_len,
            'params': emb_stats['non_emb_params'],
            'macs_attn': 0,
            'emb_macs': emb_stats['emb_macs'],
            'emb_params': emb_stats['emb_params']
        }

        for layer_idx, layer in enumerate(self.layers):
            enc_macs_params = layer.compute_macs_params(S=src_len)
            layer_name = 'Enc'
            encoder_mac_params['{}_Layer_{}'.format(layer_name, layer_idx)] = {
                'macs': enc_macs_params['macs'],
                'params': enc_macs_params['params'],
                'macs_attn': enc_macs_params['macs_attn'],
                'emb_macs': 0,
                'emb_params': 0
            }

        if self.layer_norm is not None:
            encoder_mac_params['Enc_LN'] = {
                'macs_attn': 0,
                'macs': 0,
                'params': sum([p.numel() for p in self.layer_norm.parameters()]),
                'emb_macs': 0,
                'emb_params': 0
            }

        return encoder_mac_params


class DeLighTTransformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        self._future_mask = torch.empty(0)

        self.dropout = args.dropout
        self.share_input_output_embed = args.share_decoder_input_output_embed

        embed_dim = embed_tokens.embedding_dim
        self.output_embed_dim = args.delight_emb_map_dim

        self.padding_idx = embed_tokens.map_layer.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens

        self.embed_positions = (
            PositionalEmbedding(
                args.max_target_positions,
                embed_dim,
                self.padding_idx,
                learned=args.decoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

        self.positional_dropout = RecurrentDropout(p=args.pe_dropout, batch_first=True) if not args.no_token_positional_embeddings else None

        if args.delight_dec_scaling == 'block' and (args.delight_dec_min_depth == args.delight_dec_max_depth):
            args.delight_dec_scaling = 'uniform'
            args.define_dec_fixed_depth = args.delight_dec_min_depth

        self.layers = nn.ModuleList([])

        if args.delight_dec_scaling == 'uniform':
            assert args.delight_dec_min_depth == args.delight_dec_max_depth
            self.layers.extend(
                [DeLighTTransformerDecoderLayer(args,
                                                embed_dim=embed_dim,
                                                no_encoder_attn=no_encoder_attn,
                                                dextra_depth=args.delight_dec_min_depth,
                                                width_multiplier=args.delight_dec_width_mult)
                 for _ in range(args.delight_dec_layers)]
            )
        else:
            assert args.delight_dec_min_depth < args.delight_dec_max_depth

            dextra_depths = np.linspace(start=args.delight_dec_min_depth,
                                         stop=args.delight_dec_max_depth,
                                         num=args.delight_dec_layers,
                                         dtype=np.int)

            depth_ratio = (args.delight_dec_max_depth * 1.0) / args.delight_dec_min_depth
            width_multipliers = np.linspace(start=args.delight_dec_width_mult,
                                      stop=args.delight_dec_width_mult + (depth_ratio - 1.0), # subtraction by 1 for max==min case
                                      num=args.delight_dec_layers,
                                      dtype=np.float
                                      )

            self.layers.extend(
                [DeLighTTransformerDecoderLayer(args=args,
                                                embed_dim=embed_dim,
                                                width_multiplier=round(width_multipliers[idx], 3),
                                                no_encoder_attn=no_encoder_attn,
                                                dextra_depth=layer_i)
                 for idx, layer_i in enumerate(dextra_depths)
                 ]
            )

        self.num_layers = len(self.layers)

        self.project_out_dim = (
            get_weight_layer(name='linear', in_features=embed_dim, out_features=self.output_embed_dim, use_bias=False)
            if embed_dim != self.output_embed_dim else None
        )

        self.adaptive_softmax = None

        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens.map_layer if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif not self.share_input_output_embed:
            self.embed_out = get_weight_layer(name='linear',
                                              in_features=self.output_embed_dim,
                                              out_features=len(dictionary),
                                              use_bias=False)

        if args.decoder_normalize_before and not getattr(
                args, "no_decoder_final_norm", False
        ):
            self.layer_norm = get_norm_layer(name=args.norm_type, out_features=embed_dim)
        else:
            self.layer_norm = None

    def forward(
            self,
            prev_output_tokens,
            encoder_out: Optional[EncoderOut] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            features_only: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
            src_lengths: Optional[Any] = None,
            return_all_hiddens: bool = False,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(
            self,
            prev_output_tokens,
            encoder_out: Optional[EncoderOut] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            *args, **kwargs
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """

        # embed positions
        positions = (
            self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
            if self.embed_positions is not None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_tokens(prev_output_tokens)

        if positions is not None:
            x += positions
            x = self.positional_dropout(x)
        else:
            x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]

        for layer in self.layers:
            x, layer_attn, _ = layer(
                x,
                encoder_out.encoder_out if encoder_out is not None else None,
                encoder_out.encoder_padding_mask if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=self.buffered_future_mask(x) if incremental_state is None else None,
            )
            inner_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}

    def output_layer(self, features):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            if self.share_input_output_embed:
                return F.linear(features, self.embed_tokens.map_layer.weight)
            else:
                return self.embed_out(features)
        else:
            return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (self._future_mask.size(0) == 0 or (not self._future_mask.device == tensor.device)  or self._future_mask.size(0) < dim ):
            self._future_mask = torch.triu( utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1 )
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)

        for i in range(self.num_layers):
            # update layer norms
            layer_norm_map = {
                "0": "self_attn_layer_norm",
                "1": "encoder_attn_layer_norm",
                "2": "final_layer_norm",
            }
            for old, new in layer_norm_map.items():
                for m in ("weight", "bias"):
                    k = "{}.layers.{}.layer_norms.{}.{}".format(name, i, old, m)
                    if k in state_dict:
                        state_dict[
                            "{}.layers.{}.{}.{}".format(name, i, new, m)
                        ] = state_dict[k]
                        del state_dict[k]

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) <= 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])

        return state_dict

    def compute_macs_params(self, src_len=1, tgt_len=1):
        decoder_mac_params = dict()

        emb_stats = self.embed_tokens.compute_macs_params()

        decoder_mac_params['Dec_LUT'] = {
            'macs': emb_stats['non_emb_macs'] * tgt_len,
            'params': emb_stats['non_emb_params'],
            'macs_attn': 0,
            'emb_macs': emb_stats['emb_macs'],
            'emb_params': emb_stats['emb_params']
        }

        for layer_idx, layer in enumerate(self.layers):
            dec_macs_params = layer.compute_macs_params(S=src_len, T=tgt_len)
            layer_name = 'Dec'
            decoder_mac_params['{}_Layer_{}'.format(layer_name, layer_idx)] = {
                'macs': dec_macs_params['macs'],
                'params': dec_macs_params['params'],
                'macs_attn': dec_macs_params['macs_attn'],
                'emb_macs': 0,
                'emb_params': 0
            }

        if self.layer_norm is not None:
            decoder_mac_params['Dec_LN'] = {
                'macs_attn': 0,
                'macs': 0,
                'params': sum([p.numel() for p in self.layer_norm.parameters()]),
                'emb_macs': 0,
                'emb_params': 0
            }

        if self.project_out_dim is not None:
            proj_stats = self.project_out_dim.compute_macs_params()
            decoder_mac_params['Dec_Proj'] = {
                'macs': proj_stats['macs'] * tgt_len,
                'params': proj_stats['params'],
                'macs_attn': 0,
                'emb_macs': 0,
                'emb_params': 0
            }

        # Ouptut layer
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            params_lin = self.embed_tokens.map_layer.weight.numel()
            if self.share_input_output_embed:
                decoder_mac_params['Dec_Out'] = {
                    'macs': params_lin * tgt_len,
                    'params': 0,
                    'macs_attn': 0,
                    'emb_macs': 0,
                    'emb_params': 0
                }
            else:
                decoder_mac_params['Dec_Out'] = {
                    'macs': params_lin * tgt_len,
                    'params': params_lin,
                    'macs_attn': 0,
                    'emb_macs': 0,
                    'emb_params': 0
                }
        else:
            macs_params_adapt_sm = self.adaptive_softmax.compute_macs_params()
            decoder_mac_params['Dec_Adp_Out'] = {
                'macs': macs_params_adapt_sm['macs'] * tgt_len,
                'params': macs_params_adapt_sm['params'],
                'macs_attn': 0,
                'emb_macs': 0,
                'emb_params': 0
            }

        return decoder_mac_params


@register_model_architecture("delight_transformer", "delight_transformer")
def base_architecture(args):
    # DeLighT Embedding layer
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.delight_emb_map_dim = getattr(args, "delight_emb_map_dim", 128)
    args.delight_emb_out_dim = getattr(args, "delight_emb_out_dim", 128)
    # compute the max groups in GLT
    assert args.delight_emb_out_dim % MIN_ELEMENTS_PER_GROUP == 0, 'remainder({}, {}) should be equal to 0'.format(
        args.delight_emb_out_dim, MIN_ELEMENTS_PER_GROUP)
    max_groups = 2 ** math.ceil(math.log(args.delight_emb_out_dim // MIN_ELEMENTS_PER_GROUP, 2))

    args.delight_emb_max_groups = getattr(args, "delight_emb_max_groups", max_groups)
    args.delight_emb_dropout = getattr(args, "delight_emb_dropout", DEFAULT_DROPOUT)
    args.delight_emb_depth = getattr(args, "delight_emb_depth", DEFAULT_MIN_DEXTRA_LAYERS)
    args.delight_emb_width_mult = getattr(args, "delight_emb_width_mult", DEFAULT_WIDTH_MULTIPLIER)

    # Encoder arguments in DeLighT
    args.delight_enc_scaling = getattr(args, "delight_enc_scaling", 'block')
    args.delight_enc_layers = getattr(args, "delight_enc_layers", DEFAULT_MAX_DEXTRA_LAYERS)
    args.delight_enc_min_depth = getattr(args, "delight_enc_min_depth", DEFAULT_MIN_DEXTRA_LAYERS)
    args.delight_enc_max_depth = getattr(args, "delight_enc_max_depth", DEFAULT_MAX_DEXTRA_LAYERS)
    args.delight_enc_width_mult = getattr(args, "delight_enc_width_mult", DEFAULT_WIDTH_MULTIPLIER)
    args.delight_enc_ffn_red = getattr(args, "delight_enc_ffn_red", DEFAULT_FFN_RED_FACTOR)
    args.delight_enc_max_groups = getattr(args, "delight_enc_max_groups", max_groups)

    # Decoder arguments in DeLighT
    args.delight_dec_scaling = getattr(args, "delight_dec_scaling", 'block')
    args.delight_dec_layers = getattr(args, "delight_dec_layers", DEFAULT_MAX_DEXTRA_LAYERS)
    args.delight_dec_min_depth = getattr(args, "delight_dec_min_depth", DEFAULT_MIN_DEXTRA_LAYERS)
    args.delight_dec_max_depth = getattr(args, "delight_dec_max_depth", DEFAULT_MAX_DEXTRA_LAYERS)
    args.delight_dec_width_mult = getattr(args, "delight_dec_width_mult", DEFAULT_WIDTH_MULTIPLIER)
    args.delight_dec_ffn_red = getattr(args, "delight_dec_ffn_red", DEFAULT_FFN_RED_FACTOR)
    args.delight_dec_max_groups = getattr(args, "delight_dec_max_groups", max_groups)

    ## Others
    args.no_glt_shuffle = getattr(args, "no_glt_shuffle", False)
    args.glt_shuffle = not args.no_glt_shuffle
    args.define_iclr = getattr(args, "define_iclr", False)
    args.delight_dropout = getattr(args, "delight_dropout", DEFAULT_DROPOUT)

    # normalization and activation layers
    args.norm_type = getattr(args, "norm_type", 'ln')
    args.act_type = getattr(args, "act_type", 'swish')

    # ADAPTIVE INPUT AND OUTPUT PARAMS
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.adaptive_softmax_factor = getattr(args, 'adaptive_softmax_factor', 4)
    args.tie_adaptive_weights = getattr(args, 'tie_adaptive_weights', False)
    args.tie_adaptive_proj = getattr(args, 'tie_adaptive_proj', False)

    # Print  stats
    args.print_stats = getattr(args, "print_stats", False)
    args.src_len_ps = getattr(args, "src_len_ps", 20)
    args.tgt_len_ps = getattr(args, "tgt_len_ps", 20)

    # DROPOUTS
    args.attention_dropout = getattr(args, "attention_dropout", DEFAULT_DROPOUT)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.dropout = getattr(args, "dropout", DEFAULT_DROPOUT)
    args.delight_dropout = getattr(args, "delight_dropout", 0.0)
    args.pe_dropout = getattr(args, "pe_dropout", DEFAULT_DROPOUT)
    args.ffn_dropout = getattr(args, "ffn_dropout", DEFAULT_DROPOUT)

    # Other parameters
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)

    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)


@register_model_architecture("delight_transformer", "delight_transformer_wmt14_en_de")
@register_model_architecture("delight_transformer", "delight_transformer_wmt14_de_en")
def delight_transformer_wmt14_en_de(args):
    # Training symmetric encoder-decoder

    # LUT and model dimensions
    args.delight_emb_map_dim = getattr(args, "delight_emb_map_dim", 128)
    args.delight_emb_out_dim = getattr(args, "delight_emb_out_dim", 512)

    # minimum depth
    args.delight_dec_min_depth = getattr(args, 'delight_dec_min_depth', 4)
    args.delight_enc_min_depth = getattr(args, 'delight_enc_min_depth', 4)
    #assert args.delight_enc_min_depth == args.delight_dec_min_depth, '{} != {}'.format(args.delight_enc_min_depth,
    #                                                                                   args.delight_dec_min_depth)
    args.delight_emb_depth = args.delight_dec_min_depth

    # maximum depth
    args.delight_dec_max_depth = getattr(args, 'delight_dec_max_depth', 8)
    args.delight_enc_max_depth = getattr(args, 'delight_enc_max_depth', 8)
    #assert args.delight_enc_max_depth == args.delight_dec_max_depth, '{} != {}'.format(args.delight_enc_max_depth,
    #                                                                                   args.delight_dec_max_depth)

    # we set number of encoder and decoder blocks equal to max depth
    args.delight_dec_layers = args.delight_dec_max_depth
    args.delight_enc_layers = args.delight_enc_max_depth

    # width multipliers
    args.delight_dec_width_mult = getattr(args, 'delight_dec_width_mult', 2)
    args.delight_enc_width_mult = getattr(args, 'delight_enc_width_mult', 2)
    assert args.delight_dec_width_mult == args.delight_enc_width_mult
    args.delight_emb_width_mult = args.delight_enc_width_mult

    # dropouts
    # We scale dropout values based on Transformer model. This might not be an optimal configuration and searching for
    # a good dropout values might lead to a better solution.
    scale_dropout = 0.3
    delta_model_dimension = 1024.0 / args.delight_emb_out_dim
    scale_dropout_d_m = round(scale_dropout / delta_model_dimension, 2)
    scale_dropout_d_m = bound_function(0, 0.3, scale_dropout_d_m)

    scale_attn_drop = 0.1
    scale_attn_drop_d_m = round(scale_attn_drop / delta_model_dimension, 2)
    scale_attn_drop_d_m = bound_function(0, 0.1, scale_attn_drop_d_m)

    args.dropout = getattr(args, "dropout", scale_dropout_d_m)
    args.delight_emb_dropout = getattr(args, "delight_emb_dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", scale_attn_drop_d_m)
    args.delight_dropout = getattr(args, "delight_dropout", 0.0) # we don't use this in sufficiently large datasets
    args.pe_dropout = getattr(args, "pe_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0) # We don't use it
    args.ffn_dropout = getattr(args, "ffn_dropout", scale_dropout_d_m)

    # shared embeddings
    args.share_all_embeddings = getattr(args, "share_all_embeddings", True)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", True
    )

    base_architecture(args)

@register_model_architecture("delight_transformer", "delight_transformer_wmt16_en_ro")
@register_model_architecture("delight_transformer", "delight_transformer_wmt16_ro_en")
def delight_transformer_wmt16_en_ro(args):
    # Training symmetric encoder-decoder

    # LUT and model dimensions
    args.delight_emb_map_dim = getattr(args, "delight_emb_map_dim", 128)
    args.delight_emb_out_dim = getattr(args, "delight_emb_out_dim", 512)

    # minimum depth
    args.delight_dec_min_depth = getattr(args, 'delight_dec_min_depth', 4)
    args.delight_enc_min_depth = getattr(args, 'delight_enc_min_depth', 4)
    #assert args.delight_enc_min_depth == args.delight_dec_min_depth, '{} != {}'.format(args.delight_enc_min_depth,
    #                                                                                   args.delight_dec_min_depth)
    args.delight_emb_depth = args.delight_dec_min_depth

    # maximum depth
    args.delight_dec_max_depth = getattr(args, 'delight_dec_max_depth', 8)
    args.delight_enc_max_depth = getattr(args, 'delight_enc_max_depth', 8)
    #assert args.delight_enc_max_depth == args.delight_dec_max_depth, '{} != {}'.format(args.delight_enc_max_depth,
    #                                                                                   args.delight_dec_max_depth)

    # we set number of encoder and decoder blocks equal to max depth
    args.delight_dec_layers = args.delight_dec_max_depth
    args.delight_enc_layers = args.delight_enc_max_depth

    # width multipliers
    args.delight_dec_width_mult = getattr(args, 'delight_dec_width_mult', 2)
    args.delight_enc_width_mult = getattr(args, 'delight_enc_width_mult', 2)
    assert args.delight_dec_width_mult == args.delight_enc_width_mult
    args.delight_emb_width_mult = args.delight_enc_width_mult

    # dropouts
    # We scale dropout values based on Transformer model. This might not be an optimal configuration and searching for
    # a good dropout values might lead to a better solution.
    scale_dropout = 0.3
    delta_model_dimension = 1024.0 / args.delight_emb_out_dim
    scale_dropout_d_m = round(scale_dropout / delta_model_dimension, 2)
    scale_dropout_d_m = bound_function(0, 0.3, scale_dropout_d_m)

    scale_attn_drop = 0.1
    scale_attn_drop_d_m = round(scale_attn_drop / delta_model_dimension, 2)
    scale_attn_drop_d_m = bound_function(0, 0.1, scale_attn_drop_d_m)

    scale_delight_drop = 0.1
    delta_dim = 512.0 / args.delight_emb_out_dim
    scale_delight_drop_d_m = round(scale_delight_drop / delta_dim, 2)
    scale_delight_drop_d_m = bound_function(0, 0.1, scale_delight_drop_d_m)

    args.dropout = getattr(args, "dropout", scale_dropout_d_m)
    args.delight_emb_dropout = getattr(args, "delight_emb_dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", scale_attn_drop_d_m)
    args.delight_dropout = getattr(args, "delight_dropout", scale_delight_drop_d_m)
    args.pe_dropout = getattr(args, "pe_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0) # we don't use it
    args.ffn_dropout = getattr(args, "ffn_dropout", scale_dropout_d_m)

    # shared embeddings
    args.share_all_embeddings = getattr(args, "share_all_embeddings", True)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", True
    )

    base_architecture(args)

@register_model_architecture("delight_transformer", "delight_transformer_wmt14_en_fr")
@register_model_architecture("delight_transformer", "delight_transformer_wmt14_fr_en")
def delight_transformer_wmt16_en_de(args):
    # Training symmetric encoder-decoder

    # LUT and model dimensions
    args.delight_emb_map_dim = getattr(args, "delight_emb_map_dim", 128)
    args.delight_emb_out_dim = getattr(args, "delight_emb_out_dim", 512)

    # minimum depth
    args.delight_dec_min_depth = getattr(args, 'delight_dec_min_depth', 4)
    args.delight_enc_min_depth = getattr(args, 'delight_enc_min_depth', 4)
    #assert args.delight_enc_min_depth == args.delight_dec_min_depth, '{} != {}'.format(args.delight_enc_min_depth,
    #                                                                                   args.delight_dec_min_depth)
    args.delight_emb_depth = args.delight_dec_min_depth

    # maximum depth
    args.delight_dec_max_depth = getattr(args, 'delight_dec_max_depth', 8)
    args.delight_enc_max_depth = getattr(args, 'delight_enc_max_depth', 8)
    #assert args.delight_enc_max_depth == args.delight_dec_max_depth, '{} != {}'.format(args.delight_enc_max_depth,
    #                                                                                   args.delight_dec_max_depth)

    # we set number of encoder and decoder blocks equal to max depth
    args.delight_dec_layers = args.delight_dec_max_depth
    args.delight_enc_layers = args.delight_enc_max_depth

    # width multipliers
    args.delight_dec_width_mult = getattr(args, 'delight_dec_width_mult', 2)
    args.delight_enc_width_mult = getattr(args, 'delight_enc_width_mult', 2)
    assert args.delight_dec_width_mult == args.delight_enc_width_mult
    args.delight_emb_width_mult = args.delight_enc_width_mult

    # dropouts
    # We scale dropout values based on Transformer model. This might not be an optimal configuration and searching for
    # a good dropout values might lead to a better solution.
    scale_dropout = 0.1
    delta_model_dimension = 1024.0 / args.delight_emb_out_dim
    scale_dropout_d_m = round(scale_dropout / delta_model_dimension, 2)
    scale_dropout_d_m = bound_function(0, 0.1, scale_dropout_d_m)

    scale_attn_drop = 0.1
    scale_attn_drop_d_m = round(scale_attn_drop / delta_model_dimension, 2)
    scale_attn_drop_d_m = bound_function(0, 0.1, scale_attn_drop_d_m)

    args.dropout = getattr(args, "dropout", scale_dropout_d_m)
    args.delight_emb_dropout = getattr(args, "delight_emb_dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", scale_attn_drop_d_m)
    args.delight_dropout = getattr(args, "delight_dropout", 0.0) # we don't use for large datasets
    args.pe_dropout = getattr(args, "pe_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0) # we don't use it
    args.ffn_dropout = getattr(args, "ffn_dropout", scale_dropout_d_m)

    # shared embeddings
    args.share_all_embeddings = getattr(args, "share_all_embeddings", True)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", True
    )

    base_architecture(args)


@register_model_architecture("delight_transformer", "delight_transformer_iwslt_de_en")
@register_model_architecture("delight_transformer", "delight_transformer_iwslt_en_de")
def delight_transformer_iwslt_de_en(args):
    # Training symmetric encoder-decoder

    # LUT and model dimensions
    args.delight_emb_map_dim = getattr(args, "delight_emb_map_dim", 128)
    args.delight_emb_out_dim = getattr(args, "delight_emb_out_dim", 512)

    # minimum depth
    args.delight_dec_min_depth = getattr(args, 'delight_dec_min_depth', 4)
    args.delight_enc_min_depth = getattr(args, 'delight_enc_min_depth', 4)
    #assert args.delight_enc_min_depth == args.delight_dec_min_depth, '{} != {}'.format(args.delight_enc_min_depth,
    #                                                                                   args.delight_dec_min_depth)
    args.delight_emb_depth = args.delight_dec_min_depth

    # maximum depth
    args.delight_dec_max_depth = getattr(args, 'delight_dec_max_depth', 8)
    args.delight_enc_max_depth = getattr(args, 'delight_enc_max_depth', 8)
    #assert args.delight_enc_max_depth == args.delight_dec_max_depth, '{} != {}'.format(args.delight_enc_max_depth,
    #                                                                                   args.delight_dec_max_depth)

    # we set number of encoder and decoder blocks equal to max depth
    args.delight_dec_layers = args.delight_dec_max_depth
    args.delight_enc_layers = args.delight_enc_max_depth

    # width multipliers
    args.delight_dec_width_mult = getattr(args, 'delight_dec_width_mult', 2)
    args.delight_enc_width_mult = getattr(args, 'delight_enc_width_mult', 2)
    assert args.delight_dec_width_mult == args.delight_enc_width_mult
    args.delight_emb_width_mult = args.delight_enc_width_mult

    # dropouts
    # We scale dropout values based on Transformer model. This might not be an optimal configuration and searching for
    # a good dropout values might lead to a better solution.
    scale_dropout = 0.3
    delta_model_dimension = 512.0 / args.delight_emb_out_dim
    scale_dropout_d_m = round(scale_dropout / delta_model_dimension, 2)
    scale_dropout_d_m = bound_function(0, 0.3, scale_dropout_d_m)

    scale_attn_drop = 0.1
    scale_attn_drop_d_m = round(scale_attn_drop / delta_model_dimension, 2)
    scale_attn_drop_d_m = bound_function(0, 0.1, scale_attn_drop_d_m)

    scale_delight_drop = 0.1
    delta_dim = 512.0 / args.delight_emb_out_dim
    scale_delight_drop_d_m = round(scale_delight_drop / delta_dim, 2)
    scale_delight_drop_d_m = bound_function(0, 0.1, scale_delight_drop_d_m)

    args.dropout = getattr(args, "dropout", scale_dropout_d_m)
    args.delight_emb_dropout = getattr(args, "delight_emb_dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", scale_attn_drop_d_m)
    args.delight_dropout = getattr(args, "delight_dropout", scale_delight_drop_d_m)
    args.pe_dropout = getattr(args, "pe_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0) # we don't use it
    args.ffn_dropout = getattr(args, "ffn_dropout", scale_dropout_d_m)

    # shared embeddings
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", True
    )

    base_architecture(args)
