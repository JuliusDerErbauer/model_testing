from chamfer_distance import ChamferDistance
from torch import nn
import torch


class ChamferLoss(nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()
        self.loss = ChamferDistance()

    def forward(self, x, y):
        x = x.permute(0, 2, 1)
        y = y.permute(0, 2, 1)
        loss_x = self.loss(x, y)
        print(loss_x.size())
        print(loss_x)
        (loss_y, _) = self.loss(x, y)

        return torch.mean(loss_x) + torch.mean(loss_y)
