from torch import nn
from pytorch3d.loss import chamfer_distance
from sinkhorn import SinkhornDistance


class ChamferLoss(nn.Module):

    def forward(self, x, y):
        x = x.permute(0, 2, 1)
        y = y.permute(0, 2, 1)
        dist_x, dist_y, indices_x, indices_y = chamfer_distance(x, y, point_reduction='mean')
        print(f"dist_x: {dist_x}, dist_y: {dist_y}")
        return dist_x + dist_y


class EMDLoss(nn.Module):
    def forward(self, x, y):
        x = x.permute(0, 2, 1)
        y = y.permute(0, 2, 1)
        sinkhorn = SinkhornDistance(eps=1e-3, max_iter=200, reduction='mean')
        dist, _, _ = sinkhorn(x, y)
        return dist


class CombinedLoss(nn.Module):
    def __init__(self, losses, weights):
        super(CombinedLoss, self).__init__()
        assert len(losses) == len(weights), "The number of losses and weights must be the same"
        self.losses = nn.ModuleList(losses)
        self.weights = weights

    def forward(self, x, y):
        total_loss = 0
        for loss, weight in zip(self.losses, self.weights):
            total_loss += weight * loss(x, y)
        return total_loss
