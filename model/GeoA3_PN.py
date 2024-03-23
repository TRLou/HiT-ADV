'''
Pytorch 0.3.1
'''
from __future__ import division
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models


def _get_indices_knn_T(points, k):
    r_a = torch.sum(points * points, dim=1, keepdim=True)
    dis = torch.bmm(torch.transpose(points, 1, 2), points).mul_(-2)
    dis.add_(r_a.transpose(1, 2) + r_a)

    _, indices = torch.topk(dis, k, dim=-1, largest=False, sorted=False)
    return indices


def _get_indices_knn(queries, points, k):
    r_a = torch.sum(queries * queries, dim=1, keepdim=True)
    r_b = torch.sum(points * points, dim=1, keepdim=True)
    dis = torch.bmm(torch.transpose(queries, 1, 2), points).mul_(-2)
    dis.add_(r_a.transpose(1, 2) + r_b)

    _, indices = torch.topk(dis, k, dim=-1, largest=False, sorted=True)
    return indices


def _get_distance(queries, points):
    r_a = torch.sum(queries * queries, dim=1, keepdim=True)
    r_b = torch.sum(points * points, dim=1, keepdim=True)
    dis = torch.bmm(torch.transpose(queries, 1, 2), points).mul_(-2)
    dis.add_(r_a.transpose(1, 2) + r_b)

    return dis


def _init_params(m, method='constant'):
    """
    method: xavier_uniform, kaiming_normal, constant
    """
    if isinstance(m, list):
        for im in m:
            _init_params(im, method)
    else:
        if method == 'xavier_uniform':
            nn.init.xavier_uniform_(m.weight.data)
        elif method == 'kaiming_normal':
            nn.init.kaiming_normal(m.weight.data, mode='fan_in')
        elif isinstance(method, (int, float)):
            m.weight.data.fill_(method)
        else:
            raise ValueError("unknown method.")
        if m.bias is not None:
            m.bias.data.zero_()


class transform_net(nn.Module):
    def __init__(self, K=3):
        super(transform_net, self).__init__()
        self.eps = 1e-3
        self.K = K

        self.conv1 = nn.Conv1d(K, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, K * K)

        self.bn1 = nn.BatchNorm1d(64, eps=self.eps)
        self.bn2 = nn.BatchNorm1d(128, eps=self.eps)
        self.bn3 = nn.BatchNorm1d(1024, eps=self.eps)
        self.bn4 = nn.BatchNorm1d(512, eps=self.eps)
        self.bn5 = nn.BatchNorm1d(256, eps=self.eps)

        self.relu = nn.ReLU(True)
        self._init_module()

    def forward(self, input):
        feat = self.relu(self.bn1(self.conv1(input)))
        feat = self.relu(self.bn2(self.conv2(feat)))
        feat = self.relu(self.bn3(self.conv3(feat)))
        feat, _ = torch.max(feat, -1)
        feat = self.relu(self.bn4(self.fc1(feat)))
        feat = self.relu(self.bn5(self.fc2(feat)))
        feat = self.fc3(feat)

        return feat.view(feat.size(0), self.K, self.K)

    def _init_module(self):
        _init_params([self.conv1, self.conv2, self.conv3], 'xavier_uniform')
        _init_params([self.fc1, self.fc2], 'xavier_uniform')
        _init_params([self.bn1, self.bn2, self.bn3, self.bn4, self.bn5], 1)
        self.fc3.weight.data.fill_(0)
        self.fc3.bias.data.copy_(torch.eye(self.K).view(-1).float())


class PointNet(nn.Module):
    def __init__(self, classes, return_idx=False, npoint=1024):
        super(PointNet, self).__init__()
        self.num_class = classes
        self.eps = 1e-3
        self.return_idx = return_idx

        self.input_transform = transform_net(K=3)
        self.feature_transform = transform_net(K=64)

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 3, 1, 1)

        self.bn1 = nn.BatchNorm1d(64, eps=self.eps)
        self.bn2 = nn.BatchNorm1d(64, eps=self.eps)
        self.bn3 = nn.BatchNorm1d(64, eps=self.eps)
        self.bn4 = nn.BatchNorm1d(128, eps=self.eps)
        self.bn5 = nn.BatchNorm1d(1024, eps=self.eps)

        self.fc1 = nn.Linear(1024, 512)
        self.bn6 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(256, self.num_class)

        self.relu = nn.ReLU(True)
        self.drop1 = nn.Dropout(p=0.3)
        self.drop2 = nn.Dropout(p=0.3)

        self._init_module()

    def forward(self, pc):
        batch_size = pc.size(0)
        num_point = pc.size(2)
        assert pc.size(1) == 3

        transform = self.input_transform(pc)
        feat = torch.bmm(pc.permute(0, 2, 1), transform).permute(0, 2, 1)
        feat = self.relu(self.bn1(self.conv1(feat)))
        feat = self.relu(self.bn2(self.conv2(feat)))

        transform = self.feature_transform(feat)
        feat = torch.bmm(feat.permute(0, 2, 1), transform).permute(0, 2, 1)
        feat = self.relu(self.bn3(self.conv3(feat)))
        feat = self.relu(self.bn4(self.conv4(feat)))
        feat = self.relu(self.bn5(self.conv5(feat)))
        feat, idx = feat.max(-1)

        # final MLP
        feat = self.drop1(self.relu(self.bn6(self.fc1(feat))))
        feat = self.drop2(self.relu(self.bn7(self.fc2(feat))))
        output = self.fc3(feat)

        if self.training:
            return output, transform
        else:
            if self.return_idx:
                return output, idx
            else:
                return output

    def _init_module(self):
        _init_params([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.fc1, self.fc2, self.fc3],
                     'xavier_uniform')
        _init_params([self.bn1, self.bn2, self.bn3, self.bn4, self.bn5, self.bn6, self.bn7], 1)

    def adjust_bn_momentum(self, epoch, bn_momentum):
        momentum = bn_momentum * (0.5 ** (epoch // 20))
        momentum = max(momentum, 0.01)

        def _set_bn_momentum(m, momentum):
            if isinstance(m, list):
                for im in m:
                    _set_bn_momentum(im, momentum)
            else:
                m.momentum = momentum

        _set_bn_momentum([self.bn1, self.bn2, self.bn3, self.bn4, self.bn5, self.bn6, self.bn7], momentum)
        _set_bn_momentum(
            [self.input_transform.bn1, self.input_transform.bn2, self.input_transform.bn3, self.input_transform.bn4,
             self.input_transform.bn5], momentum)
        _set_bn_momentum([self.feature_transform.bn1, self.feature_transform.bn2, self.feature_transform.bn3,
                          self.feature_transform.bn4, self.feature_transform.bn5], momentum)



























