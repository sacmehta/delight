# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import operator
import functools

import torch
import torch.nn.functional as F
from torch import nn


class TiedLinear(nn.Module):
    def __init__(self, weight, transpose):
        super().__init__()
        self.weight = weight
        self.transpose = transpose

    def forward(self, input):
        return F.linear(input, self.weight.t() if self.transpose else self.weight)

    def compute_macs_params(self):
        macs = self.weight.numel()
        return {
            'name': self.__class__.__name__,
            'macs': macs,
            'params': 0,  # since tied
        }


class TiedHeadModule(nn.Module):
    def __init__(self, weights, input_dim, num_classes):
        super().__init__()
        tied_emb, _ = weights
        self.num_words, emb_dim = tied_emb.size()

        self.word_proj = TiedLinear(tied_emb, transpose=False)
        if input_dim != emb_dim:
            self.word_proj = nn.Sequential(
                nn.Linear(input_dim, emb_dim, bias=False),
                self.word_proj,
            )

        self.class_proj = nn.Linear(input_dim, num_classes, bias=False)
        self.out_dim = self.num_words + num_classes

        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    def forward(self, input):
        inp_sz = functools.reduce(operator.mul, input.shape[:-1], 1)
        out = self._float_tensor.new(inp_sz, self.out_dim)
        out[:, :self.num_words] = self.word_proj(input.view(inp_sz, -1))
        out[:, self.num_words:] = self.class_proj(input.view(inp_sz, -1))
        return out

    def compute_macs_params(self):
        macs = 0
        params = 0

        # word Projection
        if isinstance(self.word_proj, nn.Sequential):
            params_lin = sum([p.numel() for p in self.word_proj[0].parameters()])
            macs_lin = params_lin
            macs_params_wp = self.word_proj[1].compute_macs_params()

            macs += macs_params_wp['macs'] + macs_lin
            params += macs_params_wp['params'] + params_lin
        else:
            macs_params_wp = self.word_proj.compute_macs_params()
            macs += macs_params_wp['macs']
            params += macs_params_wp['params']

        # class projection
        params_cp = sum([p.numel() for p in self.class_proj.parameters()])
        macs += params_cp  # MACS and PArams are same for Linear..
        params += params_cp

        return {
            'name': self.__class__.__name__,
            'macs': macs,
            'params': params,
        }


class AdaptiveSoftmax(nn.Module):
    """
    This is an implementation of the efficient softmax approximation for
    graphical processing units (GPU), described in the paper "Efficient softmax
    approximation for GPUs" (http://arxiv.org/abs/1609.04309).
    """

    def __init__(self, vocab_size, input_dim, cutoff, dropout, factor=4., adaptive_inputs=None, tie_proj=False):
        super().__init__()

        if vocab_size > cutoff[-1]:
            cutoff = cutoff + [vocab_size]
        else:
            assert vocab_size == cutoff[
                -1], 'cannot specify cutoff larger than vocab size'

        output_dim = cutoff[0] + len(cutoff) - 1

        self.vocab_size = vocab_size
        self.cutoff = cutoff
        self.dropout = dropout
        self.input_dim = input_dim
        self.factor = factor

        self.lsm = nn.LogSoftmax(dim=1)

        if adaptive_inputs is not None:
            self.head = TiedHeadModule(adaptive_inputs.weights_for_band(0), input_dim, len(cutoff) - 1)
        else:
            self.head = nn.Linear(input_dim, output_dim, bias=False)

        self._make_tail(adaptive_inputs, tie_proj)

        def init_weights(m):
            if hasattr(m, 'weight') and not isinstance(m, TiedLinear) and not isinstance(m, TiedHeadModule):
                nn.init.xavier_uniform_(m.weight)

        self.apply(init_weights)

        self.register_buffer('version', torch.LongTensor([1]))

    def _make_tail(self, adaptive_inputs=None, tie_proj=False):
        self.tail = nn.ModuleList()
        for i in range(len(self.cutoff) - 1):
            dim = int(self.input_dim // self.factor ** (i + 1))

            tied_emb, tied_proj = adaptive_inputs.weights_for_band(i + 1) \
                if adaptive_inputs is not None else (None, None)

            if tied_proj is not None:
                if tie_proj:
                    proj = TiedLinear(tied_proj, transpose=True)
                else:
                    proj = nn.Linear(tied_proj.size(0), tied_proj.size(1), bias=False)
            else:
                proj = nn.Linear(self.input_dim, dim, bias=False)

            m = nn.Sequential(
                proj,
                nn.Dropout(self.dropout),
                nn.Linear(
                    dim, self.cutoff[i + 1] - self.cutoff[i], bias=False,
                ) if tied_emb is None else TiedLinear(tied_emb, transpose=False),
            )

            self.tail.append(m)

    def upgrade_state_dict_named(self, state_dict, name):
        version_name = name + '.version'
        if version_name not in state_dict:
            raise Exception('This version of the model is no longer supported')

    def adapt_target(self, target):
        """
        In order to be efficient, the AdaptiveSoftMax does not compute the
        scores for all the word of the vocabulary for all the examples. It is
        thus necessary to call the method adapt_target of the AdaptiveSoftMax
        layer inside each forward pass.
        """

        target = target.view(-1)
        new_target = [target.clone()]
        target_idxs = []

        for i in range(len(self.cutoff) - 1):
            mask = target.ge(self.cutoff[i]).mul(target.lt(self.cutoff[i + 1]))
            new_target[0][mask] = self.cutoff[0] + i

            if mask.any():
                target_idxs.append(mask.nonzero().squeeze(1))
                new_target.append(target[mask].add(-self.cutoff[i]))
            else:
                target_idxs.append(None)
                new_target.append(None)

        return new_target, target_idxs

    def forward(self, input, target):
        """
        Args:
            input: (b x t x d)
            target: (b x t)
        Returns:
            2 lists: output for each cutoff section and new targets by cut off
        """

        input = input.contiguous().view(-1, input.size(-1))
        input = F.dropout(input, p=self.dropout, training=self.training)

        new_target, target_idxs = self.adapt_target(target)
        output = [self.head(input)]

        for i in range(len(target_idxs)):
            if target_idxs[i] is not None:
                output.append(self.tail[i](input.index_select(0, target_idxs[i])))
            else:
                output.append(None)

        return output, new_target

    def get_log_prob(self, input, target):
        """
        Computes the log probabilities for all the words of the vocabulary,
        given a 2D tensor of hidden vectors.
        """

        bsz, length, dim = input.size()
        input = input.contiguous().view(-1, dim)

        if target is not None:
            _, target_idxs = self.adapt_target(target)
        else:
            target_idxs = None

        head_y = self.head(input)
        log_probs = head_y.new_zeros(input.size(0), self.vocab_size)

        head_sz = self.cutoff[0] + len(self.tail)
        log_probs[:, :head_sz] = self.lsm(head_y)
        tail_priors = log_probs[:, self.cutoff[0]: head_sz].clone()

        for i in range(len(self.tail)):
            start = self.cutoff[i]
            end = self.cutoff[i + 1]

            if target_idxs is None:
                tail_out = log_probs[:, start:end]
                tail_out.copy_(self.tail[i](input))
                log_probs[:, start:end] = self.lsm(tail_out).add_(tail_priors[:, i, None])
            elif target_idxs[i] is not None:
                idxs = target_idxs[i]
                tail_out = log_probs[idxs, start:end]
                tail_out.copy_(self.tail[i](input[idxs]))
                log_probs[idxs, start:end] = self.lsm(tail_out).add_(tail_priors[idxs, i, None])

        log_probs = log_probs.view(bsz, length, -1)
        return log_probs

    def compute_macs_params(self):
        macs = 0
        n_params = 0

        # compute for head
        if isinstance(self.head, TiedHeadModule):
            macs_params_head = self.head.compute_macs_params()

            macs += macs_params_head['macs']
            n_params += macs_params_head['params']
        else:
            params_head = sum([p.numel() for p in self.head.parameters()])
            macs += params_head
            n_params += params_head

        # compute for tail

        for m in self.tail.modules():
            if isinstance(m, TiedLinear):
                macs_params_tail_m = m.compute_macs_params()

                macs += macs_params_tail_m['macs']
                n_params += macs_params_tail_m['params']
            elif isinstance(m, nn.Linear):
                params_tail_m = sum([p.numel() for p in m.parameters()])

                macs += params_tail_m
                n_params += params_tail_m

        return {
            'name': self.__class__.__name__,
            'macs': macs,
            'params': n_params
        }