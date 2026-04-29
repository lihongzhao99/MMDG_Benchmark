import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, pretrained=False, in_channel=1, num_classes=10):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=32, stride=1),  # 8, 20449, 32
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=1),  # 8, 20448, 31
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(8, 16, kernel_size=8, stride=1),  # 16, 20441, 24
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=1),  # 16, 20440, 23
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, stride=1),  # 32, 20438, 21
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=1),  # 32, 20437, 20
        )

        self.layer4 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=3, stride=1),  # 32, 20435, 18
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(4)  # 32, 4, 4
        )


    def forward(self, x):

        x = x.unsqueeze(1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)



        x = x.view(x.size(0), -1)

        return x


class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x, alpha=1.0):
    return GradientReversalLayer.apply(x, alpha)


class Encoder(nn.Module):
    def __init__(self, input_dim=256, out_dim=6, hidden=256):
        super().__init__()
        self.enc_net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, feat):
        return self.enc_net(feat)


class ProjectHead(nn.Module):
    def __init__(self, input_dim, out_dim=128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3)
        )

    def forward(self, x):
        return self.proj(x)


class DomainClassifier(nn.Module):
    def __init__(self, input_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class GMP(nn.Module):

    def __init__(self, num_classes=6, num_domains=3, proj_dim=128):
        super().__init__()

        self.vib_net = CNN()
        self.aud_net = CNN()

        # 模态分类头
        self.cls_v = nn.Linear(128, num_classes)
        self.cls_a = nn.Linear(128, num_classes)

        # 融合分类头（与 train_GMP 的 mlp_cls 对齐）
        self.cls_fusion = Encoder(input_dim=256, out_dim=num_classes, hidden=256)

        # 局部领域对抗头（与 train_GMP 的 *_proj2 + domain_disc 对齐）
        self.v_proj_domain = ProjectHead(128, proj_dim)
        self.a_proj_domain = ProjectHead(128, proj_dim)
        self.domain_disc_local = DomainClassifier(proj_dim, out_dim=num_domains)

    def forward(self, vibration, sound):
        vib_feat = self.vib_net(vibration)
        aud_feat = self.aud_net(sound)
        fusion_feat = torch.cat((vib_feat, aud_feat), dim=1)
        return self.cls_fusion(fusion_feat)

    def forward_train(self, vibration, sound, alpha_domain=0.3):
        vib_feat = self.vib_net(vibration)
        aud_feat = self.aud_net(sound)
        fusion_feat = torch.cat((vib_feat, aud_feat), dim=1)

        # 分类输出
        v_logit = self.cls_v(vib_feat)
        a_logit = self.cls_a(aud_feat)
        fusion_logit = self.cls_fusion(fusion_feat)

        # 局部领域对抗（GRL）
        v_domain_feat = self.v_proj_domain(grad_reverse(vib_feat, alpha_domain))
        a_domain_feat = self.a_proj_domain(grad_reverse(aud_feat, alpha_domain))
        v_domain_logit = self.domain_disc_local(v_domain_feat)
        a_domain_logit = self.domain_disc_local(a_domain_feat)

        return {
            'v_logit': v_logit,
            'a_logit': a_logit,
            'fusion_logit': fusion_logit,
            'v_domain_logit': v_domain_logit,
            'a_domain_logit': a_domain_logit,
            'v_feat': vib_feat,
            'a_feat': aud_feat,
        }






