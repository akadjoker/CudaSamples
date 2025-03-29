import torch
import torch.nn as nn
from torchvision.models import resnet18

class ResNetSteering(nn.Module):
    def __init__(self, freeze_features=True):
        super().__init__()
        base = resnet18(pretrained=True)
        print(base)

        if freeze_features:
            for param in base.parameters():
                param.requires_grad = False

        # Substituir a última camada para regressão
        base.fc = nn.Sequential(
            nn.Linear(base.fc.in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.model = base

    def forward(self, x):
        return self.model(x).squeeze()

