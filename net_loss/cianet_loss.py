import numpy as np
from torch import nn

from .loss_utils import DeepSupervisionWrapper, SegLoss


class CIANetLoss(nn.Module):
    def __init__(self, sup_depth):
        super().__init__()
        self.seg_loss = SegLoss(sup_depth)

        weights = np.array([1 / (2 ** i) for i in range(2)])
        weights = weights / weights.sum()
        self.bnd_loss = DeepSupervisionWrapper(nn.BCELoss(), weights)
        self.sigmoid = nn.Sigmoid()

        print('CIANetLoss initialized')

    def forward(self, net_out, target):
        seg_tgt = [i[:, 0:1] for i in target]
        s_loss = self.seg_loss([net_out[0], seg_tgt])

        deep_size = min(len(net_out[1] if len(net_out) > 1 else []), len(target))
        if deep_size == 0:
            b_loss = 0
        else:
            out2 = [self.sigmoid(net_out[1][i]) for i in range(deep_size)]
            tgt2 = [target[i][:, 1:2] for i in range(deep_size)]
            b_loss = self.bnd_loss(out2, tgt2)

        return s_loss + 0.1 * b_loss
