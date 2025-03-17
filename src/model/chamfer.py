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
        dist_x = self.loss(x, y)
        dist_y = self.loss(y, x)

        print("dist_x:", dist_x)
        print("dist_y:", dist_y)

        return torch.mean(dist_x) + torch.mean(dist_y)
