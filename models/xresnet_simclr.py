from fastai.vision.all import *
import torch

class xresnet18SimCLR(nn.Module):
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.backbone = xresnet18(ndim=1, c_in=12, n_out=out_dim)
        dim_mlp = self.backbone[-1].in_features
        # add mlp projection head
        self.backbone[-1] = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone[-1])

    def forward(self, x): # [batch_size, 12, 2500] -> [batch_size, out_dim]
        return self.backbone(x)

class xresnet34SimCLR(nn.Module):
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.backbone = xresnet34(ndim=1, c_in=12, n_out=out_dim)
        dim_mlp = self.backbone[-1].in_features
        # add mlp projection head
        self.backbone[-1] = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone[-1])

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
