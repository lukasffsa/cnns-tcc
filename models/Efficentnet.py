import torch.nn as nn
from torchvision import models

class EfficientNetSingleRegressor(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(EfficientNetSingleRegressor, self).__init__()
        
        # Carrega a EfficientNet-B0 com os melhores pesos disponíveis no PyTorch
        self.efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        
        # 1. Congela todo o extrator de características (features)
        for param in self.efficientnet.parameters():
            param.requires_grad = False
            
        # 2. Descongela os últimos blocos convolucionais (estágios 7 e 8)
        # Equivalente ao que fizemos na layer4 da ResNet50
        for param in self.efficientnet.features[7:].parameters():
            param.requires_grad = True
            
        # 3. Substitui o cabeçote de classificação por uma regressão contínua
        in_features = self.efficientnet.classifier[1].in_features  # 1280 no caso da B0
        
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 1)
        )

    def forward(self, x):
        # Retorna o tensor achatado com shape [batch_size]
        return self.efficientnet(x).squeeze(-1)