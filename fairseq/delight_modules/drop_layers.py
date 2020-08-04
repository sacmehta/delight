# ============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
# ============================================

import torch
import torch.nn as nn


class RecurrentDropout(nn.Module):
    '''
        Applies the same dropout mask across all time steps
    '''

    def __init__(self, p, batch_first=False):
        '''
        :param p: Dropout probability
        :param batch_first: Batch first or not
        '''
        super().__init__()
        self.p = p
        self.keep_p = 1.0 - p
        self.batch_first = batch_first

    def forward(self, x):
        '''
        :param x: Input of dimension [B x T x C] (batch first) or [T x B x C]
        :return: output of dimension [B x T x C] (batch first) or [T x B x C]
        '''
        if not self.training:
            return x

        if self.p <= 0 or self.p >= 1:
            return x

        assert x.dim() == 3, 'Input should be [B x T x C] or [T x B x C]'

        # recurrent dropout
        if self.batch_first:
            #  x --> B x T x C
            m = x.new_empty(x.size(0), 1, x.size(2), requires_grad=False).bernoulli_(self.keep_p)
        else:
            # x --> T x B x C
            m = x.new_empty(1, x.size(1), x.size(2), requires_grad=False).bernoulli_(self.keep_p)
        m = m.div_(self.keep_p)
        m = m.expand_as(x)
        return m * x

    def __repr__(self):
        s = '{name}(p={p})'
        return s.format(name=self.__class__.__name__, **self.__dict__)
