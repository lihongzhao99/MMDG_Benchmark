
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


class ProjectionHead(nn.Module):
    """Projection head used by contrastive objectives (L_C, L_UNC)."""
    def __init__(self, input_dim=128, hidden_dim=256, out_dim=64):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, feat):
        return F.normalize(self.head(feat), dim=1)


class NEL(nn.Module):

    def __init__(self, num_classes=6, proj_hidden_dim=256, proj_out_dim=64):
        super(NEL, self).__init__()

        self.sharedNet1 = CNN()
        self.sharedNet2 = CNN()
        self.fusion_cls = nn.Linear(256, num_classes)
        self.vib_proj = ProjectionHead(
            input_dim=128, hidden_dim=proj_hidden_dim, out_dim=proj_out_dim
        )
        self.aud_proj = ProjectionHead(
            input_dim=128, hidden_dim=proj_hidden_dim, out_dim=proj_out_dim
        )

    def extract_features(self, vibration, sound):
        vibration_feature = self.sharedNet1(vibration)
        sound_feature = self.sharedNet2(sound)
        combined_features = torch.cat((vibration_feature, sound_feature), dim=1)
        return vibration_feature, sound_feature, combined_features

    def project_embeddings(self, vibration_feature, sound_feature):
        vibration_proj = self.vib_proj(vibration_feature)
        sound_proj = self.aud_proj(sound_feature)
        return vibration_proj, sound_proj

    def forward(self, vibration, sound, flag=0, return_features=False):
        vibration_feature, sound_feature, combined_features = self.extract_features(vibration, sound)
        pred3 = self.fusion_cls(combined_features)

        if not return_features:
            return pred3

        vibration_proj, sound_proj = self.project_embeddings(vibration_feature, sound_feature)

        return pred3, vibration_feature, sound_feature, vibration_proj, sound_proj, combined_features






