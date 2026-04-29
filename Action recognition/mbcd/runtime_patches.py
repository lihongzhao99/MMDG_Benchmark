"""Process-local patches that reproduce MBCD's framework extensions.

The original MBCD tree edited mmaction and VGGSound classes directly.  This
module applies the same behavioral additions only inside the MBCD training
process, so existing methods in Others keep using their current framework code.
"""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F


def _reparameterize(mu, std):
    epsilon = torch.randn_like(std)
    return mu + epsilon * std


def _sqrtvar(x, eps):
    value = (x.var(dim=0, keepdim=True) + eps).sqrt()
    return value.repeat(x.shape[0], 1)


def _dsu_3d_tensor(self, x, eps=1e-6, p=0.5):
    if (not self.training) or (random.random() > p):
        return x

    mean = x.mean(dim=[2, 3, 4], keepdim=False)
    std = (x.var(dim=[2, 3, 4], keepdim=False) + eps).sqrt()

    beta = _reparameterize(mean, _sqrtvar(mean, eps))
    gamma = _reparameterize(std, _sqrtvar(std, eps))

    x = (x - mean.reshape(x.shape[0], x.shape[1], 1, 1, 1)) / std.reshape(
        x.shape[0], x.shape[1], 1, 1, 1)
    x = x * gamma.reshape(x.shape[0], x.shape[1], 1, 1, 1) + beta.reshape(
        x.shape[0], x.shape[1], 1, 1, 1)
    return x


def _dsu_2d_tensor(self, x, eps=1e-6, p=0.5):
    if (not self.training) or (random.random() > p):
        return x

    mean = x.mean(dim=[2, 3], keepdim=False)
    std = (x.var(dim=[2, 3], keepdim=False) + eps).sqrt()

    beta = _reparameterize(mean, _sqrtvar(mean, eps))
    gamma = _reparameterize(std, _sqrtvar(std, eps))

    x = (x - mean.reshape(x.shape[0], x.shape[1], 1, 1)) / std.reshape(
        x.shape[0], x.shape[1], 1, 1)
    x = x * gamma.reshape(x.shape[0], x.shape[1], 1, 1) + beta.reshape(
        x.shape[0], x.shape[1], 1, 1)
    return x


def _patch_mmaction_backbones():
    from mmaction.models.backbones.resnet3d import ResNet3d
    from mmaction.models.backbones.resnet3d_slowfast import ResNet3dSlowFast

    def resnet3d_avg_pool(self, x):
        x = F.adaptive_avg_pool3d(x, (1, 1, 1))
        return x.view(x.size(0), -1)

    def slowfast_avg_pool(self, x):
        x_fast, x_slow = x
        x_fast = F.adaptive_avg_pool3d(x_fast, (1, 1, 1))
        x_slow = F.adaptive_avg_pool3d(x_slow, (1, 1, 1))
        x = torch.cat((x_slow, x_fast), dim=1)
        return x.view(x.size(0), -1)

    def slowfast_dsu(self, x, eps=1e-6, p=0.5):
        if (not self.training) or (random.random() > p):
            return x

        x_slow = x[0]
        x_fast = x[1]

        mean1 = x_slow.mean(dim=[2, 3, 4], keepdim=False)
        std1 = (x_slow.var(dim=[2, 3, 4], keepdim=False) + eps).sqrt()
        mean2 = x_fast.mean(dim=[2, 3, 4], keepdim=False)
        std2 = (x_fast.var(dim=[2, 3, 4], keepdim=False) + eps).sqrt()

        beta1 = _reparameterize(mean1, _sqrtvar(mean1, eps))
        gamma1 = _reparameterize(std1, _sqrtvar(std1, eps))
        beta2 = _reparameterize(mean2, _sqrtvar(mean2, eps))
        gamma2 = _reparameterize(std2, _sqrtvar(std2, eps))

        x_slow = (
            x_slow - mean1.reshape(x_slow.shape[0], x_slow.shape[1], 1, 1, 1)
        ) / std1.reshape(x_slow.shape[0], x_slow.shape[1], 1, 1, 1)
        x_slow = (
            x_slow * gamma1.reshape(x_slow.shape[0], x_slow.shape[1], 1, 1, 1)
            + beta1.reshape(x_slow.shape[0], x_slow.shape[1], 1, 1, 1)
        )
        x_fast = (
            x_fast - mean2.reshape(x_fast.shape[0], x_fast.shape[1], 1, 1, 1)
        ) / std2.reshape(x_fast.shape[0], x_fast.shape[1], 1, 1, 1)
        x_fast = (
            x_fast * gamma2.reshape(x_fast.shape[0], x_fast.shape[1], 1, 1, 1)
            + beta2.reshape(x_fast.shape[0], x_fast.shape[1], 1, 1, 1)
        )

        return (x_slow, x_fast)

    ResNet3d.avg_pool = resnet3d_avg_pool
    ResNet3d.dsu = _dsu_3d_tensor
    ResNet3dSlowFast.avg_pool = slowfast_avg_pool
    ResNet3dSlowFast.dsu = slowfast_dsu


def _patch_mmaction_heads():
    from mmaction.models.heads.base import BaseHead
    from mmaction.models.heads.i3d_head import I3DHead
    from mmaction.models.heads.slowfast_head import SlowFastHead

    def slowfast_init(self,
                      num_classes,
                      in_channels,
                      loss_cls=dict(type='CrossEntropyLoss'),
                      spatial_type='avg',
                      dropout_ratio=0.8,
                      init_std=0.01,
                      reduce_channel=False,
                      reduce_channel_num=512,
                      **kwargs):
        BaseHead.__init__(self, num_classes, in_channels, loss_cls, **kwargs)
        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        self.reduce_channel = reduce_channel
        self.reduce_channel_num = reduce_channel_num

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

        if self.reduce_channel:
            print("reduce_channel")
            self.fc_reduce = nn.Linear(in_channels, self.reduce_channel_num)
            self.fc_cls = nn.Linear(self.reduce_channel_num, num_classes)
        else:
            self.fc_cls = nn.Linear(in_channels, num_classes)

        if self.spatial_type == 'avg':
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = None

    def get_pred(self, x):
        return self.fc_cls(x)

    SlowFastHead.__init__ = slowfast_init
    SlowFastHead.get_pred = get_pred
    I3DHead.get_pred = get_pred


def _patch_vggsound_resnet():
    from VGGSound.models.resnet import ResNet

    def get_feature(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

    def get_predict(self, x):
        return self.layer4(x)

    def avg_pool(self, x):
        x = self.avgpool(x)
        return x.view(x.size(0), -1)

    def get_pred(self, x):
        return self.fc(x)

    def cls_head(self, x):
        x = self.avgpool(x)
        x_ = x.reshape(x.size(0), -1)
        x = self.fc(x_)
        return x, x_

    ResNet.get_feature = get_feature
    ResNet.get_predict = get_predict
    ResNet.avg_pool = avg_pool
    ResNet.get_pred = get_pred
    ResNet.cls_head = cls_head
    ResNet.dsu = _dsu_2d_tensor


def apply_mbcd_runtime_patches():
    """Apply MBCD-only mmaction/VGGSound extensions in the current process."""
    _patch_mmaction_backbones()
    _patch_mmaction_heads()
    _patch_vggsound_resnet()

