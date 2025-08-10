import torch
import torch.nn as nn
from torchvision import models

class ImageEncoder(nn.Module):
    def __init__(self, name: str = "resnet18", pretrained: bool = True, out_dim: int = 128):
        super().__init__()
        if name == "resnet18":
            backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            feat_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.backbone = backbone
        else:
            raise ValueError(f"Unsupported image encoder: {name}")
        self.proj = nn.Linear(feat_dim, out_dim)

    def forward(self, x):
        feats = self.backbone(x)
        return self.proj(feats)
