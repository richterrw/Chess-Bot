"""A small residual convolutional network that predicts a human's move.

Input: 21 board planes (see nn_data.encode) plus the mover's time control,
rating and an "is Robbie" flag broadcast as constant planes. Output: one
logit per (from-square, to-square) pair, 4096 in all, in the mover's frame.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn_data import N_PLANES

N_EXTRA = 5  # 3 time-control one-hot planes, rating, is-Robbie


def norm_rating(elo):
    return (elo - 1100.0) / 300.0


class Res(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)


class MoveNet(nn.Module):
    def __init__(self, ch=64, blocks=6):
        super().__init__()
        self.ch, self.n_blocks = ch, blocks
        self.stem = nn.Conv2d(N_PLANES + N_EXTRA, ch, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(ch)
        self.blocks = nn.ModuleList([Res(ch) for _ in range(blocks)])
        self.head = nn.Conv2d(ch, 64, 1)

    def forward(self, planes, tc, rating, robbie):
        b = planes.shape[0]
        extra = torch.cat([F.one_hot(tc, 3).float(), rating[:, None], robbie[:, None]], dim=1)
        x = torch.cat([planes, extra[:, :, None, None].expand(b, N_EXTRA, 8, 8)], dim=1)
        x = F.relu(self.stem_bn(self.stem(x)))
        for blk in self.blocks:
            x = blk(x)
        out = self.head(x)                       # [B, to(64), 8, 8 = from]
        return out.flatten(2).transpose(1, 2).reshape(b, 4096)  # index from*64 + to
