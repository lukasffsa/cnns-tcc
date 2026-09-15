import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image


class AlfalfaDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.DataFrame(pd.read_csv(csv_file, sep=';'), columns=['ALTURA', 'Cultivar', '1', '2', '3', '4', '5', 'Média', 'PARCELA', 'K2O (kg/ha)', 'MS (kg/ha)'])

        self.data.fillna(0, inplace=True) 
        
        # --- CALCULANDO OS VALORES MÁXIMOS PARA NORMALIZAÇÃO ---
        self.max_altura = self.data['Média'].max()
        self.max_biomassa = self.data['MS (kg/ha)'].max()
        
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]
        self.id_col = self.data.columns[0]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        basename = os.path.splitext(img_name)[0]
        partes = basename.split('_')
        parcela = partes[0][:4] 
        mes = partes[1]
        
        csv_id = (parcela + mes).upper()
        row = self.data[self.data[self.id_col] == csv_id]
        
        if row.empty:
            raise ValueError(f"ID {csv_id} não encontrado no CSV.")
            
        # --- APLICANDO A NORMALIZAÇÃO [0, 1] ---
        altura_real = row['Média'].values[0]
        biomassa_real = row['MS (kg/ha)'].values[0]
        
        altura_norm = altura_real / self.max_altura
        biomassa_norm = biomassa_real / self.max_biomassa
        
        targets = torch.tensor([altura_norm, biomassa_norm], dtype=torch.float32)

        return image, targets