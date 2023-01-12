import torch
import torch.nn as nn


class MayoCNNECG6(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 3), stride=(1, 1), padding='same', bias=True),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 3), stride=(1, 1), padding='same', bias=True),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 3), stride=(1, 1), padding='same', bias=True),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 5), stride=(1, 1), padding='same', bias=True),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 5), stride=(1, 1), padding='same', bias=True),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 5), stride=(1, 1), padding='same', bias=True),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0),
        )

        self.spatial = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(12, 1), stride=(1, 1), padding=0, bias=True),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU(),
        )

        self.fc = nn.Linear(in_features=39, out_features=num_classes)

    def forward(self, x): # [batch_size, 12, 2500] -> [batch_size, 39]
        x = torch.unsqueeze(x, 1)
        t = self.temporal(x)
        s = self.spatial(t)
        s = s.flatten(start_dim=1)
        o = self.fc(s)
        return o

class MayoCNNECG6SimCLR(nn.Module):
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.backbone = MayoCNNECG6(num_classes=out_dim)
        dim_mlp = self.backbone.fc.in_features
        # add mlp projection head
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

    def forward(self, x): # [batch_size, 12, 2500] -> [batch_size, out_dim]
        return self.backbone(x)


if __name__ == '__main__':
    x = torch.randn(32, 12, 2500)
    model = MayoCNNECG6SimCLR()
    print(model)
    y = model(x)
    print(y.shape)
