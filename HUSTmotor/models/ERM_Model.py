
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

class ERM(nn.Module):

    def __init__(self, num_classes=6):
        super(ERM, self).__init__()

        self.sharedNet1 = CNN()
        self.sharedNet2 = CNN()
        self.fusion_cls = nn.Linear(256, num_classes)

    def forward(self, vibration, sound, flag=0):


        vibration_feature = self.sharedNet1(vibration)
        sound_feature = self.sharedNet2(sound)

        combined_features = torch.cat((vibration_feature, sound_feature), dim=1)
        pred3 = self.fusion_cls(combined_features)

        return pred3






