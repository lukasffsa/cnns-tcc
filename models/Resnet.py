import torch.nn as nn
from torchvision import models

class ResNet50SingleRegressor(nn.Module):
    def __init__(self):
        super(ResNet50SingleRegressor, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Congela as camadas base (extrator de características genéricas)
        for param in self.resnet.parameters():
            param.requires_grad = False
            
        # Descongela a layer4 para adaptação ao domínio agrícola
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True
            
        # Camada final com 1 única saída
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        return self.resnet(x).squeeze(-1)  