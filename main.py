import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sklearn.model_selection

# Configuración
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 64
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
NUM_CLASSES = 9  # Número de tipos de fractales

"""
1. tree
2. koch
3. dragon
4. sierpinski_carpet
5. julia
6. sierpinski
7. newton
8. barnsley
9. mandelbrot
"""

# Dataset personalizado
class FractalDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe
        self.transform = transform
        self.label_map = {label: idx for idx, label in enumerate(sorted(self.data['label'].unique()))}  # Asignar índices

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['relative_path']
        label = self.data.iloc[idx]['fractal']
        
        # Leer imagen
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype(np.float32) / 255.0  # Normalización [0, 1]
        img = np.expand_dims(img, axis=0)  # Añadir canal
        
        label_idx = self.label_map[label]  # Convertir etiqueta a índice
        return torch.tensor(img, dtype=torch.float32), torch.tensor(label_idx, dtype=torch.long)

# Modelo CNN
class FractalCNN(nn.Module):
    def __init__(self):
        super(FractalCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(out_features=256), nn.ReLU(),
            nn.Linear(256, NUM_CLASSES)  # Salida con 9 neuronas
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x  # Sin Softmax porque CrossEntropyLoss lo incluye

# Cargar datos
# CSV con columnas: 'image_path' y 'label'
df = pd.read_csv('fractathlon/train.csv', sep=',')
train_df, val_df = sklearn.model_selection.train_test_split(df, test_size=0.2, random_state=42)

train_dataset = FractalDataset(train_df)
val_dataset = FractalDataset(val_df)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Entrenamiento
model = FractalCNN().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

train_losses, val_losses = [], []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)
    
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    val_accuracy = 100 * correct / total
    
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val Acc: {val_accuracy:.4f}%")
    print("-" * 50)

# Visualizar pérdidas
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.savefig("fractal_loss.png", bbox_inches='tight')