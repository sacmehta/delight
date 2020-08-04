# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torch import nn
import math
from typing import List


class AdaptiveInput(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        padding_idx: int,
        initial_dim: int,
        factor: float,
        output_dim: int,
        cutoff: List[int],
        no_scale_emb: bool = False
    ):
        super().__init__()

        if vocab_size > cutoff[-1]:
            cutoff = cutoff + [vocab_size]
        else:
            assert vocab_size == cutoff[
                -1], 'cannot specify cutoff larger than vocab size'

        self.cutoff = cutoff
        self.embedding_dim = output_dim
        self.padding_idx = padding_idx

        self.embeddings = nn.ModuleList()
        self.projections = nn.ModuleList()
        self.embed_scales = []
        self.padding_idxes = []
        for i in range(len(self.cutoff)):
            prev = self.cutoff[i - 1] if i > 0 else 0
            size = self.cutoff[i] - prev
            dim = int(initial_dim // (factor ** i))

            emb = nn.Embedding(size, dim, self.padding_idx)

            # initialize embedding layer
            nn.init.normal_(emb.weight, mean=0, std=emb.weight.shape[1] ** -0.5)
            nn.init.constant_(emb.weight[self.padding_idx], 0)

            proj = nn.Linear(dim, output_dim, bias=False)
            # initialize Linear Layer
            nn.init.xavier_uniform_(proj.weight)

            self.embeddings.append(emb)
            self.projections.append(proj)

            self.padding_idx = None
            self.embed_scales.append(1.0 if no_scale_emb else math.sqrt(dim))
        self.padding_idx = padding_idx

        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    def weights_for_band(self, band: int):
        return self.embeddings[band].weight, self.projections[band].weight

    def forward(self, input: torch.Tensor):
        result = self._float_tensor.new(input.shape + (self.embedding_dim,))
        for i in range(len(self.cutoff)):
            mask = input.lt(self.cutoff[i])
            if i > 0:
                mask.mul_(input.ge(self.cutoff[i - 1]))
                chunk_input = input[mask] - self.cutoff[i - 1]
            else:
                chunk_input = input[mask]
            if mask.any():
                scaled_emb = self.embeddings[i](chunk_input) * self.embed_scales[i] # embedding layer
                scaled_emb = self.projections[i](scaled_emb) # projection layer
                result[mask] = scaled_emb
        return result

    def __repr__(self):
        s = '{name}(cutoff={cutoff}, output_features={embedding_dim})'
        for e, p in zip(self.embeddings, self.projections):
            s += '\n \t \t {} --> {}'.format(e, p)
        s += '\n'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def compute_macs_params(self):
        embedding_macs = 0
        embedding_params = 0

        for m in self.embeddings:
            embedding_params += sum([p.numel() for p in m.parameters()])
            # LUT does not have any MACs
            embedding_macs += 0

        proj_macs = 0
        proj_params = 0
        for m in self.projections:
            n_params_lin = sum([p.numel() for p in m.parameters()])
            proj_macs += n_params_lin
            proj_params += n_params_lin

        return {
            'name': self.__class__.__name__,
            'proj_macs': proj_macs,
            'proj_params': proj_params,
            'embedding_params': embedding_params,
            'embedding_macs': embedding_macs
        }


if __name__ == '__main__':
    inp = AdaptiveInput(vocab_size=100000, padding_idx=1, initial_dim=256, factor=2, output_dim=128, cutoff=[40000, 80000])

    for m in inp.embeddings:
        print(m)

    for m in inp.projections:
        print(m)