import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, pretrained=False, in_channel=1, num_classes=10):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=32, stride=1),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=1),
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(8, 16, kernel_size=8, stride=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=1),
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, stride=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=1),
        )

        self.layer4 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=3, stride=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(4),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x.view(x.size(0), -1)


class Encoder(nn.Module):
    def __init__(self, input_dim=256, out_dim=6, hidden=256):
        super().__init__()
        self.enc_net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, feat):
        return self.enc_net(feat)


class EncoderTrans(nn.Module):
    def __init__(self, input_dim=256, out_dim=256, hidden=256):
        super().__init__()
        self.enc_net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, feat):
        return self.enc_net(feat)


class ProjectHead(nn.Module):
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


class SimMMDG(nn.Module):
    def __init__(
        self,
        num_classes=6,
        backbone_dim=128,
        embedding_dim=256,
        cls_hidden_dim=256,
        trans_hidden_dim=256,
        proj_hidden_dim=256,
        proj_out_dim=64,
    ):
        super().__init__()

        self.sharedNet1 = CNN()
        self.sharedNet2 = CNN()

        self.vib_encoder = Encoder(
            input_dim=backbone_dim,
            out_dim=embedding_dim,
            hidden=cls_hidden_dim,
        )
        self.aud_encoder = Encoder(
            input_dim=backbone_dim,
            out_dim=embedding_dim,
            hidden=cls_hidden_dim,
        )
        self.fusion_cls = Encoder(
            input_dim=embedding_dim * 2,
            out_dim=num_classes,
            hidden=cls_hidden_dim,
        )

        proj_input_dim = embedding_dim // 2
        self.vib_proj = ProjectHead(
            input_dim=proj_input_dim,
            hidden_dim=proj_hidden_dim,
            out_dim=proj_out_dim,
        )
        self.aud_proj = ProjectHead(
            input_dim=proj_input_dim,
            hidden_dim=proj_hidden_dim,
            out_dim=proj_out_dim,
        )

        self.vib_to_aud = EncoderTrans(
            input_dim=embedding_dim,
            out_dim=embedding_dim,
            hidden=trans_hidden_dim,
        )
        self.aud_to_vib = EncoderTrans(
            input_dim=embedding_dim,
            out_dim=embedding_dim,
            hidden=trans_hidden_dim,
        )

    def extract_backbone_features(self, vibration, sound):
        vibration_feature = self.sharedNet1(vibration)
        sound_feature = self.sharedNet2(sound)
        return vibration_feature, sound_feature

    def encode_modalities(self, vibration_feature, sound_feature):
        vibration_embedding = self.vib_encoder(vibration_feature)
        sound_embedding = self.aud_encoder(sound_feature)
        return vibration_embedding, sound_embedding

    def project_embeddings(self, vibration_embedding, sound_embedding):
        half_dim = vibration_embedding.size(1) // 2
        vibration_proj = self.vib_proj(vibration_embedding[:, :half_dim])
        sound_proj = self.aud_proj(sound_embedding[:, :half_dim])
        return vibration_proj, sound_proj

    def forward(self, vibration, sound, flag=0, return_features=False):
        vibration_feature, sound_feature = self.extract_backbone_features(vibration, sound)
        vibration_embedding, sound_embedding = self.encode_modalities(
            vibration_feature, sound_feature
        )
        fused_embedding = torch.cat((vibration_embedding, sound_embedding), dim=1)
        pred = self.fusion_cls(fused_embedding)

        if not return_features:
            return pred

        vibration_proj, sound_proj = self.project_embeddings(
            vibration_embedding, sound_embedding
        )
        return (
            pred,
            vibration_embedding,
            sound_embedding,
            vibration_proj,
            sound_proj,
            fused_embedding,
        )

