import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms
from sklearn.model_selection import KFold

from Dataset import AlfalfaDataset
from models.Resnet import ResNet50SingleRegressor
from models.Efficentnet import EfficientNetSingleRegressor
from models.Mobilenet import MobileNetSingleRegressor
from views import plot_separated_kfold_results

# Wrapper para isolar os transforms de treino e teste
class DatasetWrapper(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        image, targets = self.subset[index]
        if self.transform:
            image = self.transform(image)
        return image, targets

    def __len__(self):
        return len(self.subset)

# Transformações com augmentation moderado para treino
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Inicialização dos dados
base_dataset = AlfalfaDataset(csv_file='dataset/metadata.csv', img_dir='dataset/images/', transform=None)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Executando em: {device}")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
epochs = 50

# Acumuladores de validação out-of-fold
all_y_true_alt, all_y_pred_alt = [], []
all_y_true_bio, all_y_pred_bio = [], []

def train_target_model(model, train_loader, target_idx, epochs, lr=1e-4):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    
    for epoch in range(epochs):
        model.train()
        for images, targets in train_loader:
            images = images.to(device)
            target = targets[:, target_idx].to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
    return model

# Loop do 5-Fold Cross-Validation
for fold, (train_idx, test_idx) in enumerate(kf.split(base_dataset)):
    print(f"\n{'='*45}")
    print(f"INICIANDO FOLD {fold+1}/5")
    print(f"{'='*45}")

    train_subset = Subset(base_dataset, train_idx)
    test_subset = Subset(base_dataset, test_idx)

    train_dataset = DatasetWrapper(train_subset, transform=train_transform)
    test_dataset = DatasetWrapper(test_subset, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # 1. Treinamento da Rede Dedicada de Altura (Target Index 0)
    print("Treinando modelo de ALTURA (MobileNetV3)...")
    model_alt = MobileNetSingleRegressor().to(device)
    model_alt = train_target_model(model_alt, train_loader, target_idx=0, epochs=epochs)

    # 2. Treinamento da Rede Dedicada de Biomassa (Target Index 1)
    print("Treinando modelo de BIOMASSA (MobileNetV3)...")
    model_bio = MobileNetSingleRegressor().to(device)
    model_bio = train_target_model(model_bio, train_loader, target_idx=1, epochs=epochs)

    # 3. Avaliação no conjunto de teste da dobra
    model_alt.eval()
    model_bio.eval()

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            
            pred_alt_norm = model_alt(images).cpu().numpy()
            pred_bio_norm = model_bio(images).cpu().numpy()
            
            true_alt_norm = targets[:, 0].numpy()
            true_bio_norm = targets[:, 1].numpy()

            # Desfaz a normalização
            all_y_true_alt.extend(true_alt_norm * base_dataset.max_altura)
            all_y_pred_alt.extend(pred_alt_norm * base_dataset.max_altura)
            
            all_y_true_bio.extend(true_bio_norm * base_dataset.max_biomassa)
            all_y_pred_bio.extend(pred_bio_norm * base_dataset.max_biomassa)

    print(f"Fold {fold+1} concluído.")

# Visualização e Métricas Finais
plot_separated_kfold_results(all_y_true_alt, all_y_pred_alt, all_y_true_bio, all_y_pred_bio)