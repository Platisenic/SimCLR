from fastai.vision.all import *
import torch

class xresnet18Libos(nn.Module):
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.res = xresnet18(ndim=1, c_in=12, n_out=out_dim)
        self.fc = self.res[-1]
        self.res[-1] = nn.Identity()

    def forward(self, x): # [batch_size, 12, 2500] -> [batch_size, out_dim]
        x = self.res(x)
        x = self.fc(x)
        return x

class xresnet18SimCLR(nn.Module):
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.backbone = xresnet18Libos(out_dim=out_dim)
        dim_mlp = self.backbone.fc.in_features
        # add mlp projection head
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

    def forward(self, x): # [batch_size, 12, 2500] -> [batch_size, out_dim]
        return self.backbone(x)


if __name__ == '__main__':
    model = xresnet18SimCLR()
    print(model)
    print(f'Total Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    x = torch.randn(64, 12, 2500)
    y = model(x)
    print(x.shape)
    print(y.shape)
