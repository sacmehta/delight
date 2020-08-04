# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq import options, utils
from fairseq.models import (
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import (
    Embedding,
    TransformerDecoder,
)
from fairseq.modules import (
    AdaptiveInput,
    CharacterTokenEmbedder,
)
from fairseq.distributed_utils import is_master

DEFAULT_MAX_TARGET_POSITIONS = 1024


@register_model('transformer_lm')
class TransformerLanguageModel(FairseqLanguageModel):

    @classmethod
    def hub_models(cls):

        def moses_fastbpe(path):
            return {
                'path': path,
                'tokenizer': 'moses',
                'bpe': 'fastbpe',
            }

        return {
            'transformer_lm.gbw.adaptive_huge': 'https://dl.fbaipublicfiles.com/fairseq/models/lm/adaptive_lm_gbw_huge.tar.bz2',
            'transformer_lm.wiki103.adaptive': 'https://dl.fbaipublicfiles.com/fairseq/models/lm/adaptive_lm_wiki103.tar.bz2',
            'transformer_lm.wmt19.en': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt19.en.tar.bz2'),
            'transformer_lm.wmt19.de': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt19.de.tar.bz2'),
            'transformer_lm.wmt19.ru': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt19.ru.tar.bz2'),
        }

    def __init__(self, decoder):
        super().__init__(decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-output-dim', type=int, metavar='N',
                            help='decoder output dimension')
        parser.add_argument('--decoder-input-dim', type=int, metavar='N',
                            help='decoder input dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--no-decoder-final-norm', action='store_true',
                            help='don\'t add an extra layernorm after the last decoder block')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion')
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        parser.add_argument('--adaptive-softmax-factor', type=float, metavar='N',
                            help='adaptive input factor')
        parser.add_argument('--no-token-positional-embeddings', action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--character-embeddings', action='store_true',
                            help='if set, uses character embedding convolutions to produce token embeddings')
        parser.add_argument('--character-filters', type=str, metavar='LIST',
                            default='[(1, 64), (2, 128), (3, 192), (4, 256), (5, 256), (6, 256), (7, 256)]',
                            help='size of character embeddings')
        parser.add_argument('--character-embedding-dim', default=4, type=int, metavar='N',
                            help='size of character embeddings')
        parser.add_argument('--char-embedder-highway-layers', default=2, type=int, metavar='N',
                            help='number of highway layers for character token embeddder')
        parser.add_argument('--adaptive-input', action='store_true',
                            help='if set, uses adaptive input')
        parser.add_argument('--adaptive-input-factor', type=float, metavar='N',
                            help='adaptive input factor')
        parser.add_argument('--adaptive-input-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive input cutoff points.')
        parser.add_argument('--tie-adaptive-weights', action='store_true',
                            help='if set, ties the weights of adaptive softmax and adaptive input')
        parser.add_argument('--tie-adaptive-proj', action='store_true',
                            help='if set, ties the projection weights of adaptive softmax and adaptive input')
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

        # Print  stats
        parser.add_argument('--print-stats', action='store_true', help='Print MACs')
        parser.add_argument('--tgt-len-ps', type=int, help='Target length for printing stats')

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

        if args.character_embeddings:
            embed_tokens = CharacterTokenEmbedder(
                task.source_dictionary, eval(args.character_filters),
                args.character_embedding_dim, args.decoder_embed_dim,
                args.char_embedder_highway_layers,
            )
        elif args.adaptive_input:
            embed_tokens = AdaptiveInput(
                len(task.source_dictionary), task.source_dictionary.pad(), args.decoder_input_dim,
                args.adaptive_input_factor, args.decoder_embed_dim,
                options.eval_str_list(args.adaptive_input_cutoff, type=int),
            )
        else:
            embed_tokens = Embedding(len(task.source_dictionary), args.decoder_input_dim, task.source_dictionary.pad())

        if args.tie_adaptive_weights:
            assert args.adaptive_input
            assert args.adaptive_input_factor == args.adaptive_softmax_factor
            assert args.adaptive_softmax_cutoff == args.adaptive_input_cutoff, '{} != {}'.format(
                args.adaptive_softmax_cutoff, args.adaptive_input_cutoff)
            assert args.decoder_input_dim == args.decoder_output_dim

        decoder = TransformerDecoder(
            args, task.target_dictionary, embed_tokens, no_encoder_attn=True,
        )

        # print macs and params layer-wise
        if args.print_stats and is_master(args):
            cls.comptue_stats(args, decoder)

        return TransformerLanguageModel(decoder)

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

                    if args.share_all_embeddings and k == 'Dec_LUT':  # Look-up Table is shared
                        macs = v['macs'] + v['emb_macs']
                        params = v['params']
                    else:
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


@register_model_architecture('transformer_lm', 'transformer_lm')
def base_lm_architecture(args):
    # backward compatibility for older model checkpoints
    if hasattr(args, 'no_tie_adaptive_proj'):
        # previous models defined --no-tie-adaptive-proj, so use the existence of
        # that option to determine if this is an "old" model checkpoint
        args.no_decoder_final_norm = True  # old models always set this to True
        if args.no_tie_adaptive_proj is False:
            args.tie_adaptive_proj = True
    if hasattr(args, 'decoder_final_norm'):
        args.no_decoder_final_norm = not args.decoder_final_norm

    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.0)

    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 2048)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.adaptive_softmax_factor = getattr(args, 'adaptive_softmax_factor', 4)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.activation_fn = getattr(args, 'activation_fn', 'relu')

    args.add_bos_token = getattr(args, 'add_bos_token', False)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.character_embeddings = getattr(args, 'character_embeddings', False)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)

    # Model training is not stable without this
    args.decoder_normalize_before = True
    args.no_decoder_final_norm = getattr(args, 'no_decoder_final_norm', False)

    args.adaptive_input = getattr(args, 'adaptive_input', False)
    args.adaptive_input_factor = getattr(args, 'adaptive_input_factor', 4)
    args.adaptive_input_cutoff = getattr(args, 'adaptive_input_cutoff', None)

    args.tie_adaptive_weights = getattr(args, 'tie_adaptive_weights', False)
    args.tie_adaptive_proj = getattr(args, 'tie_adaptive_proj', False)

    args.no_scale_embedding = getattr(args, 'no_scale_embedding', False)
    args.layernorm_embedding = getattr(args, 'layernorm_embedding', False)

    # Print  stats
    args.print_stats = getattr(args, "print_stats", False)
    args.tgt_len_ps = getattr(args, "tgt_len_ps", 20)


@register_model_architecture('transformer_lm', 'transformer_lm_big')
def transformer_lm_big(args):
    args.decoder_layers = getattr(args, 'decoder_layers', 12)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 4096)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
    base_lm_architecture(args)


@register_model_architecture('transformer_lm', 'transformer_lm_wiki103')
@register_model_architecture('transformer_lm', 'transformer_lm_baevski_wiki103')
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, 'decoder_layers', 16)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.dropout = getattr(args, 'dropout', 0.3)
    args.adaptive_input = getattr(args, 'adaptive_input', True)
    args.tie_adaptive_weights = getattr(args, 'tie_adaptive_weights', True)
    args.adaptive_input_cutoff = getattr(args, 'adaptive_input_cutoff', '20000,60000')
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', '20000,60000')
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0.2)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.1)
    args.no_decoder_final_norm = getattr(args, 'no_decoder_final_norm', True)
    args.tie_adaptive_proj = getattr(args, 'tie_adaptive_proj', True)
    transformer_lm_big(args)


@register_model_architecture('transformer_lm', 'transformer_lm_gbw')
@register_model_architecture('transformer_lm', 'transformer_lm_baevski_gbw')
def transformer_lm_baevski_gbw(args):
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.no_decoder_final_norm = getattr(args, 'no_decoder_final_norm', True)
    transformer_lm_big(args)


@register_model_architecture('transformer_lm', 'transformer_lm_gpt')
def transformer_lm_gpt(args):
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 768)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 3072)
    args.decoder_layers = getattr(args, 'decoder_layers', 12)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 12)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.activation_fn = getattr(args, 'activation_fn', 'gelu')
    base_lm_architecture(args)


@register_model_architecture('transformer_lm', 'transformer_lm_gpt2_small')
def transformer_lm_gpt2_small(args):
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 4096)
    args.decoder_layers = getattr(args, 'decoder_layers', 24)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.activation_fn = getattr(args, 'activation_fn', 'gelu')
    base_lm_architecture(args)


@register_model_architecture('transformer_lm', 'transformer_lm_gpt2_medium')
def transformer_lm_gpt2_medium(args):
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1280)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 5120)
    args.decoder_layers = getattr(args, 'decoder_layers', 36)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 20)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.activation_fn = getattr(args, 'activation_fn', 'gelu')
    base_lm_architecture(args)


@register_model_architecture('transformer_lm', 'transformer_lm_gpt2_big')
def transformer_lm_gpt2_big(args):
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1600)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 6400)
    args.decoder_layers = getattr(args, 'decoder_layers', 48)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 25)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.activation_fn = getattr(args, 'activation_fn', 'gelu')
    base_lm_architecture(args)
