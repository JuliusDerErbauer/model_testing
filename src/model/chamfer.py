from chamfer_distance import ChamferDistance
from torch import nn


class ChamferLoss(nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()
        self.loss = ChamferDistance()

    def forward(self, x, y):
        x = x.permute(0, 2, 1)
        y = y.permute(0, 2, 1)
        loss = self.loss(x, y)[0]
        return loss.mean()
