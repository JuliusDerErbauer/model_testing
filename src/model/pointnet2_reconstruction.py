import torch
import torch.nn as nn
import torch.nn.functional as F
from model.pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation


class PointNetGenerator(nn.Module):
    def __init__(self, num_point, up_ratio=4, bradius=1.0, use_bn=False,
                 use_ibn=False, use_normal=False, bn_decay=None):
        super(PointNetGenerator, self).__init__()
        self.num_point = num_point
        self.up_ratio = up_ratio
        self.bradius = bradius
        self.use_bn = use_bn
        self.use_ibn = use_ibn
        self.use_normal = use_normal
        self.bn_decay = bn_decay  # Not used in PyTorch BatchNorm (see notes below)

        # Set Abstraction Layers (mimicking tf pointnet_sa_module)
        in_channel = 3 + (3 if use_normal else 0)
        self.sa1 = PointNetSetAbstraction(npoint=num_point,
                                          radius=bradius * 0.05,
                                          nsample=32,
                                          in_channel=in_channel,
                                          mlp=[32, 32, 64],
                                          group_all=False,
                                          bn=use_bn,
                                          ibn=use_ibn)
        self.sa2 = PointNetSetAbstraction(npoint=num_point // 2,
                                          radius=bradius * 0.1,
                                          nsample=32,
                                          in_channel=64 + 3,
                                          mlp=[64, 64, 128],
                                          group_all=False,
                                          bn=use_bn,
                                          ibn=use_ibn)
        self.sa3 = PointNetSetAbstraction(npoint=num_point // 4,
                                          radius=bradius * 0.2,
                                          nsample=32,
                                          in_channel=128 + 3,
                                          mlp=[128, 128, 256],
                                          group_all=False,
                                          bn=use_bn,
                                          ibn=use_ibn)
        self.sa4 = PointNetSetAbstraction(npoint=num_point // 8,
                                          radius=bradius * 0.3,
                                          nsample=32,
                                          in_channel=256 + 3,
                                          mlp=[256, 256, 512],
                                          group_all=False,
                                          bn=use_bn,
                                          ibn=use_ibn)
        # Feature Propagation Layers (mimicking tf pointnet_fp_module)
        self.fp1 = PointNetFeaturePropagation(in_channel=512 + 256, mlp=[64])
        self.fp2 = PointNetFeaturePropagation(in_channel=256 + 128, mlp=[64])
        self.fp3 = PointNetFeaturePropagation(in_channel=128 + 64, mlp=[64])

        # Up-sampling layers:
        # We will concatenate features from:
        #   - up1 (from fp1), up2 (from fp2), up3 (from fp3),
        #   - l1_points (from sa1) and l0_xyz (input points)
        # Assuming channels: 64 + 64 + 64 + 64 + 3 = 259.
        self.up_convs = nn.ModuleList()
        for _ in range(up_ratio):
            block = nn.Sequential(
                nn.Conv2d(259, 256, kernel_size=1, bias=not use_bn),
                nn.BatchNorm2d(256) if use_bn else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 128, kernel_size=1, bias=not use_bn),
                nn.BatchNorm2d(128) if use_bn else nn.Identity(),
                nn.ReLU(inplace=True)
            )
            self.up_convs.append(block)
        # Final coordinate prediction layers
        self.fc1 = nn.Conv2d(128 * up_ratio, 64, kernel_size=1)
        self.fc2 = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, point_cloud):
        # point_cloud: (B, N, 6) if use_normal else (B, N, 3)
        B, N, _ = point_cloud.shape
        l0_xyz = point_cloud[:, :, :3]
        l0_points = point_cloud[:, :, 3:] if self.use_normal else None

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        # Feature propagation (using l0_xyz as the target for simplicity)
        up1 = self.fp1(l0_xyz, l4_xyz, None, l4_points)
        up2 = self.fp2(l0_xyz, l3_xyz, None, l3_points)
        up3 = self.fp3(l0_xyz, l2_xyz, None, l2_points)

        # Prepare feature for up-sampling
        def to_conv(x):
            return x.transpose(1, 2).unsqueeze(2)  # (B, C, 1, N)

        feat = torch.cat([to_conv(up1), to_conv(up2),
                          to_conv(up3), to_conv(l1_points),
                          to_conv(l0_xyz)], dim=1)

        up_features = [conv(feat) for conv in self.up_convs]
        up_features = torch.cat(up_features, dim=1)  # (B, 128*up_ratio, 1, N)

        x = F.relu(self.fc1(up_features))
        coord = self.fc2(x)  # (B, 3, 1, N)
        coord = coord.squeeze(2).transpose(1, 2)  # (B, N, 3)
        return coord, None
