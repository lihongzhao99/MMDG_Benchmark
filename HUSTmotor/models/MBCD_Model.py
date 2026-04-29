
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class PredHead(nn.Module):
    def __init__(self, input_dim, out_dim, hidden=128, dropout=0.5):
        super(PredHead, self).__init__()
        self.enc_net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, feat):
        return self.enc_net(feat)


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.layernorm = nn.LayerNorm(normalized_shape)

    def forward(self, feat):
        return self.layernorm(feat)

class MBCD(nn.Module):

    def __init__(self, num_classes=6, hidden_dim=128):
        super(MBCD, self).__init__()

        self.sharedNet1 = CNN()
        self.sharedNet2 = CNN()
        self.vib_norm = LayerNorm(128)
        self.aud_norm = LayerNorm(128)

        self.vib_cls = PredHead(128, num_classes, hidden=hidden_dim)
        self.aud_cls = PredHead(128, num_classes, hidden=hidden_dim)
        self.fusion_cls = PredHead(256, num_classes, hidden=hidden_dim)

    def forward(self, vibration, sound, flag=0, modality_drop_base=0.0, return_all_logits=False):

        vibration_feature = self.vib_norm(self.sharedNet1(vibration))
        sound_feature = self.aud_norm(self.sharedNet2(sound))

        vibration_pred = self.vib_cls(vibration_feature)
        sound_pred = self.aud_cls(sound_feature)

        if self.training and modality_drop_base > 0:
            with torch.no_grad():
                v_score = torch.max(torch.softmax(vibration_pred, dim=-1), dim=-1)[0].sum().detach()
                a_score = torch.max(torch.softmax(sound_pred, dim=-1), dim=-1)[0].sum().detach()

                eps = 1e-12
                v_r = v_score / (a_score + eps)
                a_r = a_score / (v_score + eps)

            bsz = vibration_feature.size(0)
            device = vibration_feature.device

            if v_r > 1:
                v_p = modality_drop_base + (1 - modality_drop_base) * torch.tanh(v_r - 1)
                v_mask = torch.bernoulli((1 - v_p) * torch.ones(bsz, device=device)).unsqueeze(1)
                vibration_feature = vibration_feature * v_mask
            if a_r > 1:
                a_p = modality_drop_base + (1 - modality_drop_base) * torch.tanh(a_r - 1)
                a_mask = torch.bernoulli((1 - a_p) * torch.ones(bsz, device=device)).unsqueeze(1)
                sound_feature = sound_feature * a_mask

        combined_features = torch.cat((vibration_feature, sound_feature), dim=1)
        pred3 = self.fusion_cls(combined_features)

        if return_all_logits:
            return {
                'fusion': pred3,
                'vibration': vibration_pred,
                'sound': sound_pred,
            }

        return pred3






