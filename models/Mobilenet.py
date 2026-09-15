import torch.nn as nn
from torchvision import models

class MobileNetSingleRegressor(nn.Module):
    def __init__(self):
        super(MobileNetSingleRegressor, self).__init__()
        # Carrega a MobileNetV3-Large com pesos padrão do ImageNet
        self.mobilenet = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        
        # 1. Congela os extratores de características iniciais
        for param in self.mobilenet.parameters():
            param.requires_grad = False
            
        # 2. Descongela os últimos 3 blocos da etapa de extração
        for param in self.mobilenet.features[-3:].parameters():
            param.requires_grad = True
            
        # 3. Substitui o classificador por uma projeção linear para regressão contínua
        # O vetor de saída do backbone convolucional possui 960 canais
        in_features = self.mobilenet.classifier[0].in_features
        self.mobilenet.classifier = nn.Sequential(
            nn.Linear(in_features, 1)
        )

    def forward(self, x):
        return self.mobilenet(x).squeeze(-1)