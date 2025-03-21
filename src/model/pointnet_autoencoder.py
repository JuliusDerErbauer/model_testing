"""
model by dhiraj inspried from Charles

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PCAutoEncoder(nn.Module):

    def __init__(self, point_dim=3, num_points=1024):
        super(PCAutoEncoder, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1, point_dim))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1))
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1))
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=1024, kernel_size=(1, 1))

        self.fc1 = nn.Linear(in_features=1024, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=1024)
        self.fc3 = nn.Linear(in_features=1024, out_features=num_points * 3)

        # batch norm
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(1024)
        self.bn4 = nn.BatchNorm1d(1024)

    def forward(self, x):
        xyz = x
        batch_size = x.shape[0]
        point_dim = x.shape[1]
        num_points = x.shape[2]

        x = x.view(batch_size, 1, num_points, point_dim)

        # encoder
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn1(self.conv2(x)))
        x = F.relu(self.bn1(self.conv3(x)))
        x = F.relu(self.bn2(self.conv4(x)))
        x = F.relu(self.bn3(self.conv5(x)))

        # do max pooling
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        # get the global embedding
        global_feat = x

        # decoder
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn4(self.fc2(x)))
        reconstructed_points = self.fc3(x)

        # do reshaping
        reconstructed_points = reconstructed_points.reshape(batch_size, point_dim, num_points)

        return reconstructed_points, global_feat
