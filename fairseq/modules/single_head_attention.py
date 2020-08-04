# ============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
# ============================================

# Some part is adapted from Multi-head attention unit in Fairseq to support incremental decoding

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from fairseq import utils
from torch import Tensor, nn
from fairseq.incremental_decoding_utils import with_incremental_state
from fairseq.delight_modules.nn_functions import get_weight_layer


@with_incremental_state
class SingleHeadAttention(nn.Module):
    """Single head attention as defined in DeLighT paper
    """

    def __init__(self, q_in_dim, kv_in_dim, proj_dim, out_dim,
                 dropout=0.0, bias=True,
                 self_attention=False, encoder_decoder_attention=False):
        '''
        :param embed_dim: Input dimension
        :param out_dim: Output dimension
        :param dropout: attention dropout
        :param bias: use bias or not
        :param self_attention: Using for self attention or not
        :param encoder_decoder_attention: Using for encoder-decoder attention or not
        :param qkv_proj: Project QKV or not. This is useful for projecting encoder output to query's dimensionality
        '''
        super(SingleHeadAttention, self).__init__()
        self.q_embed_dim = q_in_dim
        self.kv_embed_dim = kv_in_dim
        self.proj_dim = proj_dim
        self.out_dim = out_dim
        self.dropout = dropout

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        if self.self_attention:
            assert q_in_dim == kv_in_dim
            self.linear_kqv = get_weight_layer(name='linear',
                                               in_features=self.q_embed_dim,
                                               out_features=self.proj_dim,
                                               use_bias=True,
                                               gates=3
                                               )
        elif self.encoder_decoder_attention:
            self.linear_q = get_weight_layer(name='linear',
                                             in_features=self.q_embed_dim,
                                             out_features=self.proj_dim,
                                             use_bias=True,
                                             gates=1
                                             )
            self.linear_kv = get_weight_layer(name='linear',
                                              in_features=self.kv_embed_dim,
                                              out_features=self.proj_dim,
                                              use_bias=True,
                                              gates=2
                                              )
        self.scaling = self.proj_dim ** -0.5
        self.out_proj = get_weight_layer(name='linear',
                                      in_features=self.proj_dim,
                                      out_features=self.out_dim,
                                      use_bias=True
                                      )

        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def __repr__(self):
        s = '{name}(q_in_features={q_embed_dim}, kv_in_features={kv_embed_dim}, out_features={out_dim}, ' \
            'attn_dropout={dropout}, self_attention={self_attention}, ' \
            'encoder_decoder_attention={encoder_decoder_attention})'
        if self.self_attention:
            s += '\n  \t |---- KQV function: \t {}'.format(self.linear_kqv)
        elif self.encoder_decoder_attention:
            s += '\n  \t |---- KV function: \t {}'.format(self.linear_kv)
            s += '\n  \t |---- Q function: \t {}'.format(self.linear_q)
        s += '\n  \t |---- Proj: {}'.format(self.out_proj)
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(
            self,
            query,
            key_value: Optional[Tensor],
            key_padding_mask: Optional[Tensor] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            need_weights: bool = True,
            static_kv: bool = False,
            attn_mask: Optional[Tensor] = None,
            before_softmax: bool = False,
            need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, q_embed_dim = query.size()
        assert q_embed_dim == self.q_embed_dim, 'Error in {}. {} != {}'.format(self.__class__.__name__, q_embed_dim,
                                                                               self.q_embed_dim)
        assert list(query.size()) == [tgt_len, bsz, q_embed_dim]

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key_value = None
        else:
            saved_state = None

        if self.self_attention:
            q, k, v = torch.chunk(self.linear_kqv(query), chunks=3, dim=-1)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.linear_q(query)
            if key_value is None:
                k = v = None
            else:
                k, v = torch.chunk(self.linear_kv(key_value), chunks=2, dim=-1)
        else:
            raise NotImplementedError

        q *= self.scaling

        q = q.contiguous().transpose(0, 1)

        if k is not None:
            k = k.contiguous().transpose(0, 1)

        if v is not None:
            v = v.contiguous().transpose(0, 1)

        if saved_state is not None:
            # saved states are stored with shape (bsz, seq_len, head_dim)
            if "prev_key" in saved_state:
                prev_key = saved_state["prev_key"]
                assert prev_key is not None
                if static_kv:
                    k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
            if "prev_value" in saved_state:
                prev_value = saved_state["prev_value"]
                assert prev_value is not None
                if static_kv:
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)

            prev_key_padding_mask: Optional[Tensor] = None
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state["prev_key_padding_mask"]
            assert k is not None and v is not None
            key_padding_mask = SingleHeadAttention._append_prev_key_padding_mask(
                key_padding_mask=key_padding_mask,
                prev_key_padding_mask=prev_key_padding_mask,
                batch_size=bsz,
                src_len=k.size(1),
                static_kv=static_kv,
            )

            saved_state["prev_key"] = k
            saved_state["prev_value"] = v
            saved_state["prev_key_padding_mask"] = key_padding_mask
            # In this branch incremental_state is never None
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)

        assert k is not None
        src_len = k.size(1)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        # [B x T x C] x [B x C x S] --> [B x T x S]
        attn_weights = torch.bmm(q, k.transpose(1, 2))

        attn_weights = SingleHeadAttention.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # key_padding_mask --> (B x Src_len)
            # don't attend to padding symbols
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).to(torch.bool), float("-inf")
            )

        if before_softmax:
            return attn_weights, v

        # [B x T x S] --> [B x T x S]
        attn_weights_float = utils.softmax(
            attn_weights, dim=-1, onnx_trace=self.onnx_trace
        )
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = F.dropout(
            attn_weights_float.type_as(attn_weights),
            p=self.dropout,
            training=self.training,
        )
        assert v is not None
        # [B x T x S] x [B x S x F] --> [B x T x F]
        attn = torch.bmm(attn_probs, v)

        assert list(attn.size()) == [bsz, tgt_len, self.proj_dim]
        # [B x T x F] --> [T x B x F]
        attn = attn.transpose(0, 1).contiguous()

        # [T x B x F] --> [ T x B x F']
        attn = self.out_proj(attn)

        if need_weights:
            # [B x T x S] --> [T x B x S]
            attn_weights = attn_weights.transpose(1, 0)
            return attn, attn_weights
        else:
            attn_weights_tmp: Optional[Tensor] = None
            return attn, attn_weights_tmp

    @staticmethod
    def _append_prev_key_padding_mask(
            key_padding_mask: Optional[Tensor],
            prev_key_padding_mask: Optional[Tensor],
            batch_size: int,
            src_len: int,
            static_kv: bool,
    ) -> Optional[Tensor]:
        # saved key padding masks have shape (bsz, seq_len)
        if prev_key_padding_mask is not None and static_kv:
            new_key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), key_padding_mask.float()], dim=1
            )
        # During incremental decoding, as the padding token enters and
        # leaves the frame, there will be a time when prev or current
        # is None
        elif prev_key_padding_mask is not None:

            filler = torch.zeros(batch_size, src_len - prev_key_padding_mask.size(1))
            if prev_key_padding_mask.is_cuda:
                filler = filler.cuda()
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), filler.float()], dim=1
            )
        elif key_padding_mask is not None:
            filler = torch.zeros(batch_size, src_len - key_padding_mask.size(1))
            if key_padding_mask.is_cuda:
                filler = filler.cuda()
            new_key_padding_mask = torch.cat(
                [filler.float(), key_padding_mask.float()], dim=1
            )
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    def reorder_incremental_state(
            self, incremental_state: Dict[str, Dict[str, Optional[Tensor]]], new_order
    ):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def _get_input_buffer(
            self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(
            self,
            incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
            buffer: Dict[str, Optional[Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)

    def apply_sparse_mask(attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""
        items_to_add = {}
        keys_to_remove = []
        for k in state_dict.keys():
            if k.endswith(prefix + "in_proj_weight"):
                # in_proj_weight used to be q + k + v with same dimensions
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim: 2 * dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim:]

                keys_to_remove.append(k)

                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict.keys():
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][
                                                           dim: 2 * dim
                                                           ]
                    items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim:]

                    keys_to_remove.append(prefix + "in_proj_bias")

        for k in keys_to_remove:
            del state_dict[k]

        for key, value in items_to_add.items():
            state_dict[key] = value

    def compute_macs_params(self, T=1, S=1):
        macs = 0
        n_params = 0

        C = self.proj_dim

        # Scaled-dot-product macs
        # [B x T x C] x [B x C x S] --> [B x T x S]
        # multiplication-addition is counted as 1 because operations can be fused
        num_macs_kq = T * S * C
        # [B x T x S] x [B x S x C] --> [B x T x C]
        num_macs_v = T * C * S

        macs += num_macs_kq + num_macs_v

        if self.self_attention:
            assert T == S
            q_params = sum([p.numel() for p in self.linear_kqv.parameters()])

            # multiply by Seq length
            macs += (q_params * T)
            n_params += q_params
        elif self.encoder_decoder_attention:
            q_params = sum([p.numel() for p in self.linear_q.parameters()])
            kv_params = sum([p.numel() for p in self.linear_kv.parameters()])

            macs += (q_params * T) + (kv_params * S)
            n_params += q_params + kv_params
        else:
            raise NotImplementedError

        out_params = sum([p.numel() for p in self.out_proj.parameters()])
        macs += (out_params * T)
        n_params += out_params

        return {
            'name': self.__class__.__name__,
            'macs': macs,
            'params': n_params,
            'macs_attn': num_macs_kq + num_macs_v
        }
