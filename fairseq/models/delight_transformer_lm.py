# ============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
# ============================================

# Adapted from Transformer model in Fairseq

from fairseq import options, utils
from fairseq.models import (
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)

from fairseq.modules import AdaptiveInput

from fairseq.delight_modules.dextra_emb import DExTraEmb
from fairseq.models.delight_transformer import DeLighTTransformerDecoder
import torch
from fairseq.delight_modules import (
    DEFAULT_WIDTH_MULTIPLIER,
    DEFAULT_MIN_DEXTRA_LAYERS,
    MIN_ELEMENTS_PER_GROUP,
    DEFAULT_FFN_RED_FACTOR,
    DEFAULT_DROPOUT,
    DEFAULT_MAX_DEXTRA_LAYERS,
    ADAPTIVE_SCALE_FACTOR
)
import math
from fairseq.delight_modules.nn_functions import get_embedding_layer
from fairseq.delight_modules.math_utils import bound_function
from fairseq.distributed_utils import is_master

DEFAULT_MAX_TARGET_POSITIONS = 1024


@register_model('delight_transformer_lm')
class DeLighTTransformerLanguageModel(FairseqLanguageModel):

    @classmethod
    def hub_models(cls):
        return None

    def __init__(self, decoder):
        super().__init__(decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""

        # MAP LAYER IN DEFINE
        parser.add_argument('--adaptive-input', action='store_true',
                            help='Use Adaptive input or standard embedding for mapping function in DeFINE')
        parser.add_argument('--adaptive-input-factor', type=float, metavar='N',
                            help='adaptive input factor')
        parser.add_argument('--adaptive-input-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive input cutoff points.')

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

        # DEFINE EMBEDDING
        parser.add_argument('--delight-emb-map-dim', type=int,
                            help='Mapping dimension in DeLight embedding layer')
        parser.add_argument('--delight-emb-out-dim', type=int,
                            help='Output dimension of DeLight embedding layer')
        parser.add_argument('--delight-emb-max-groups', type=int,
                            help='Max. number of groups in DeLight embedding layers')
        parser.add_argument('--delight-emb-dropout', type=float, help='Dropout in DeLight embedding layers')
        parser.add_argument('--delight-emb-depth', type=int, help='Depth of DeLight unit in embedding layer')

        # DEFINE DECODER
        parser.add_argument('--delight-dec-scaling', type=str, choices=['block', 'uniform'],
                            help='Block-wise scaling or uniform')

        parser.add_argument('--delight-dec-layers', type=int, help='Number of DeLight decoder layers')
        parser.add_argument('--delight-dec-min-depth', type=int, help='Min. number of decoder layers')
        parser.add_argument('--delight-dec-max-depth', type=int, help='Max. number of decoder layers')
        parser.add_argument('--delight-dec-width-mult', type=int, help='Decoder width multiplier')
        parser.add_argument('--delight-dec-ffn-red', type=int,
                            help='Reduce FFN dims in DeLight decoder layer by this factor')
        parser.add_argument('--delight-dec-max-groups', type=int,
                            help='Max. groups in DeLight unit in the decoder')
        parser.add_argument('--no-glt-shuffle', action='store_true', help='Disable shuffling in GLT transformation.')

        parser.add_argument('--define-iclr', action='store_true', help='DeFINE unit as in ICLR paper')

        # DeLight COMMONS
        parser.add_argument('--norm-type', type=str, help='Normalization layer')
        parser.add_argument('--act-type', type=str, help='Activation function')
        parser.add_argument('--delight-dropout', type=float, help='Dropout value for DeLight layers')
        parser.add_argument('--ffn-dropout', type=float, help='Dropout after Light-weight FFN')

        # Print  stats
        parser.add_argument('--print-stats', action='store_true', help='Print MACs')
        parser.add_argument('--tgt-len-ps', type=int, help='Target length for printing stats')

        # fmt: off
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--pe-dropout', type=float, metavar='D',
                            help='dropout probability for positional encodings')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')

        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--no-decoder-final-norm', action='store_true',
                            help='don\'t add an extra layernorm after the last decoder block')

        parser.add_argument('--no-token-positional-embeddings', action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')

        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument('--decoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for decoder')
        parser.add_argument('--decoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        parser.add_argument('--layernorm-embedding', action='store_true',
                            help='add layernorm to embedding')
        parser.add_argument('--no-scale-embedding', action='store_true',
                            help='if True, dont scale embeddings')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_lm_architecture(args)

        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, 'max_target_positions', None) is None:
            args.max_target_positions = getattr(args, 'tokens_per_sample', DEFAULT_MAX_TARGET_POSITIONS)

        if args.adaptive_input:
            map_layer = AdaptiveInput(
                len(task.source_dictionary),
                task.source_dictionary.pad(),
                args.delight_emb_map_dim,
                args.adaptive_input_factor,
                args.delight_emb_map_dim,
                options.eval_str_list(args.adaptive_input_cutoff, type=int),
                no_scale_emb=args.no_scale_embedding
            )
        else:
            map_layer = get_embedding_layer(num_embeddings=len(task.source_dictionary),
                                            embedding_dim=args.delight_emb_map_dim,
                                            padding_idx=task.source_dictionary.pad())

        embed_tokens = DExTraEmb(args, map_layer=map_layer)

        if args.tie_adaptive_weights:
            assert args.adaptive_input
            assert args.adaptive_input_factor == args.adaptive_softmax_factor
            assert args.adaptive_softmax_cutoff == args.adaptive_input_cutoff, '{} != {}'.format(
                args.adaptive_softmax_cutoff, args.adaptive_input_cutoff)

        decoder = DeLighTTransformerDecoder(
            args, task.target_dictionary, embed_tokens, no_encoder_attn=True,
        )

        # print macs and params layer-wise
        if args.print_stats and is_master(args):
            cls.comptue_stats(args, decoder)

        return DeLighTTransformerLanguageModel(decoder)

    @classmethod
    def comptue_stats(cls, args, decoder):

        target_length = args.tgt_len_ps
        print('=' * 15 * target_length)
        print('{:<90} {:<20}'.format('', cls.__name__))
        print('=' * 15 * target_length)
        overall_macs = 0.0
        overall_params = 0.0
        round_places = 2

        dec_string = {}
        import csv
        with open('{}/decoder_stats_{}.csv'.format(args.save_dir, target_length), mode='w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            for time_step in range(1, target_length + 1):
                for dec_idx, (k, v) in enumerate(
                        decoder.compute_macs_params(src_len=1, tgt_len=time_step).items()):

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
            time_step_str += '| \t {:<10} '.format(round(macs, 3))
            print(time_step_str)
        overall_macs = round(float(overall_macs) / 1e6, round_places)
        overall_params = round(float(overall_params) / 1e6, round_places)
        print('-' * 15 * target_length)

        print('Total MACs for {} decoder timesteps: {} M'.format(target_length, overall_macs))
        print('Total parameters: {} M'.format(overall_params))
        print('=' * 15 * target_length)

        with open('{}/overall_stats_{}.csv'.format(args.save_dir, target_length), mode='w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(['Time steps', target_length])
            csv_writer.writerow(['Total MACs (in million)', overall_macs])
            csv_writer.writerow(['Total parameters (in million)', overall_params])



@register_model_architecture('delight_transformer_lm', 'delight_transformer_lm')
def base_lm_architecture(args):
    # Adaptive input settings
    args.adaptive_input = getattr(args, 'adaptive_input', False)
    args.adaptive_input_factor = getattr(args, 'adaptive_input_factor', ADAPTIVE_SCALE_FACTOR)
    args.adaptive_input_cutoff = getattr(args, 'adaptive_input_cutoff', None)

    # DeFINE EMBEDDINGS
    args.delight_emb_map_dim = getattr(args, "delight_emb_map_dim", 64)
    args.delight_emb_out_dim = getattr(args, "delight_emb_out_dim", 128)

    # compute the max groups in GLT
    assert args.delight_emb_out_dim % MIN_ELEMENTS_PER_GROUP == 0, 'remainder({}, {}) should be equal to 0'.format(
        args.delight_emb_out_dim, MIN_ELEMENTS_PER_GROUP)
    max_groups = 2 ** math.ceil(math.log(args.delight_emb_out_dim // MIN_ELEMENTS_PER_GROUP, 2))

    args.delight_emb_dropout = getattr(args, "delight_emb_dropout", DEFAULT_DROPOUT)
    args.delight_emb_depth = getattr(args, "delight_emb_depth", DEFAULT_MIN_DEXTRA_LAYERS)
    args.delight_emb_width_mult = getattr(args, "delight_emb_width_mult", DEFAULT_WIDTH_MULTIPLIER)
    args.delight_emb_max_groups = getattr(args, "delight_emb_max_groups", max_groups)

    # Decoder Settings
    args.delight_dec_scaling = getattr(args, "delight_dec_scaling", 'block')
    args.delight_dec_layers = getattr(args, "delight_dec_layers", DEFAULT_MAX_DEXTRA_LAYERS)
    args.delight_dec_min_depth = getattr(args, "delight_dec_min_depth", DEFAULT_MIN_DEXTRA_LAYERS)
    args.delight_dec_max_depth = getattr(args, "delight_dec_max_depth", DEFAULT_MAX_DEXTRA_LAYERS)
    args.delight_dec_width_mult = getattr(args, "delight_dec_width_mult", DEFAULT_WIDTH_MULTIPLIER)
    args.delight_dec_max_groups = getattr(args, "delight_dec_max_groups", max_groups)
    args.delight_dec_ffn_red = getattr(args, "delight_dec_ffn_red", DEFAULT_FFN_RED_FACTOR)

    # COMMON DEFINE SETTINGS
    args.no_glt_shuffle = getattr(args, "no_glt_shuffle", False)
    args.glt_shuffle = not args.no_glt_shuffle
    args.define_iclr = getattr(args, "define_iclr", False)
    args.delight_dropout = getattr(args, "delight_dropout", DEFAULT_DROPOUT)

    # normalization and activation layers
    args.norm_type = getattr(args, "norm_type", 'ln')
    args.act_type = getattr(args, "act_type", 'swish')

    # dropouts
    args.dropout = getattr(args, 'dropout', DEFAULT_DROPOUT)
    args.attention_dropout = getattr(args, 'attention_dropout', DEFAULT_DROPOUT)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.0)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", DEFAULT_DROPOUT)
    args.pe_dropout = getattr(args, "pe_dropout", DEFAULT_DROPOUT)
    args.ffn_dropout = getattr(args, "ffn_dropout", DEFAULT_DROPOUT)

    # ADAPTIVE OUTPUT Settings
    # backward compatibility for older model checkpoints
    if hasattr(args, 'no_tie_adaptive_proj'):
        # previous models defined --no-tie-adaptive-proj, so use the existence of
        # that option to determine if this is an "old" model checkpoint
        args.no_decoder_final_norm = True  # old models always set this to True
        if args.no_tie_adaptive_proj is False:
            args.tie_adaptive_proj = True
    if hasattr(args, 'decoder_final_norm'):
        args.no_decoder_final_norm = not args.decoder_final_norm

    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_factor = getattr(args, 'adaptive_softmax_factor', ADAPTIVE_SCALE_FACTOR)
    args.tie_adaptive_weights = getattr(args, 'tie_adaptive_weights', False)
    args.tie_adaptive_proj = getattr(args, 'tie_adaptive_proj', False)

    # Print  stats
    args.print_stats = getattr(args, "print_stats", False)
    args.tgt_len_ps = getattr(args, "tgt_len_ps", 20)

    # Other parameters
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.activation_fn = getattr(args, 'activation_fn', 'swish')

    args.add_bos_token = getattr(args, 'add_bos_token', False)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)

    # Model training is not stable without this
    args.decoder_normalize_before = True
    args.no_decoder_final_norm = getattr(args, 'no_decoder_final_norm', False)

    args.no_scale_embedding = getattr(args, 'no_scale_embedding', False)
    args.layernorm_embedding = getattr(args, 'layernorm_embedding', False)


@register_model_architecture('delight_transformer_lm', 'delight_transformer_lm_wiki103')
def delight_transformer_lm_wiki103(args):
    # LUT and model dimensions
    args.delight_emb_map_dim = getattr(args, "delight_emb_map_dim", 128)
    args.delight_emb_out_dim = getattr(args, "delight_emb_out_dim", 512)

    args.delight_dec_min_depth = getattr(args, 'delight_dec_min_depth', 4)
    args.delight_dec_max_depth = getattr(args, 'delight_dec_max_depth', 8)

    args.delight_emb_depth = args.delight_dec_min_depth

    # we set number of encoder and decoder blocks equal to max depth
    args.delight_dec_layers = args.delight_dec_max_depth

    # width multipliers
    args.delight_dec_width_mult = getattr(args, 'delight_dec_width_mult', 2)
    args.delight_emb_width_mult = args.delight_dec_width_mult


    # Adaptive settings
    args.adaptive_input = getattr(args, 'adaptive_input', True)
    args.tie_adaptive_weights = getattr(args, 'tie_adaptive_weights', True)
    args.adaptive_input_cutoff = getattr(args, 'adaptive_input_cutoff', '20000,60000')
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', '20000,60000')
    args.tie_adaptive_proj = getattr(args, 'tie_adaptive_proj', True)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0.1) # we use smaller value

    # Dropouts
    # We scale dropout values based on Transformer model. This might not be an optimal configuration and searching for
    # a good dropout values might lead to a better solution.

    scale_dropout = 0.1
    delta_model_dimension = 1024.0 / args.delight_emb_out_dim
    scale_dropout_d_m = round(scale_dropout / delta_model_dimension, 2)
    scale_dropout_d_m = bound_function(0, 0.3, scale_dropout_d_m)

    scale_attn_drop = 0.1
    scale_attn_drop_d_m = round(scale_attn_drop / delta_model_dimension, 2)
    scale_attn_drop_d_m = bound_function(0, 0.1, scale_attn_drop_d_m)
    
    scale_delight_drop = 0.1
    scale_delight_drop_d_m = round(scale_delight_drop / delta_model_dimension, 2)
    scale_delight_drop_d_m = bound_function(0, 0.1, scale_delight_drop_d_m)

    args.dropout = getattr(args, "dropout", scale_dropout_d_m)
    args.delight_emb_dropout = getattr(args, "delight_emb_dropout", 0.1)  # We used a fixed value
    args.attention_dropout = getattr(args, "attention_dropout", scale_attn_drop_d_m)
    args.delight_dropout = getattr(args, "delight_dropout", scale_delight_drop_d_m)
    args.pe_dropout = getattr(args, "pe_dropout", 0.1)  # We used a fixed value
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)  # we didn't use it
    args.ffn_dropout = getattr(args, "ffn_dropout", scale_dropout_d_m)

    args.no_decoder_final_norm = getattr(args, 'no_decoder_final_norm', True)
    base_lm_architecture(args)
