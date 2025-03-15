import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
import cv2
import pandas as pd
import numpy as np
import sklearn
import argparse
from torch.utils.data import Dataset, DataLoader

# Añadir el parser de argumentos
parser = argparse.ArgumentParser(description='Entrenamiento y predicción de fractales')
parser.add_argument('--model-preload', action='store_true', help='Cargar el modelo guardado si existe')
args = parser.parse_args()

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)

# Configuración
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 128
NUM_CLASSES = 9
BATCH_SIZE = 64
LEARNING_RATE = 0.001
PATIENCE = 20  # Para early stopping
NUM_EPOCHS = 60
GRAD_CLIP = 1.0  # Limitar los gradientes para evitar explosiones

class FractalDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe
        self.transform = transform
        self.label_map = {label: i for i, label in enumerate(sorted(self.data['fractal'].unique()))}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['relative_path']
        label = self.data.iloc[idx]['fractal']
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype(np.float32) / 255.0  # Normalización [0,1]
        img = np.expand_dims(img, axis=0)  # Agregar canal
        label_idx = self.label_map[label]
        
        label_idx = self.label_map[label]
        return torch.tensor(img, dtype=torch.float32), torch.tensor(label_idx, dtype=torch.long)

class FractalCNN(nn.Module):
    def __init__(self):
        super(FractalCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            # 128x128
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(32), nn.MaxPool2d(2), nn.Dropout(0.25),
            
            # 64x64
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2), nn.Dropout(0.25),
            
            # 32x32
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(128), nn.MaxPool2d(2), nn.Dropout(0.25),
            
            # 16x16
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(256), nn.MaxPool2d(2), nn.Dropout(0.25)
            # 8x8
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, NUM_CLASSES)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Cargar datos
df = pd.read_csv('train.csv')
train_df, val_df = sklearn.model_selection.train_test_split(df, test_size=0.2, random_state=42)

train_dataset = FractalDataset(train_df)
val_dataset = FractalDataset(val_df)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count())
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count())

# Configuración del modelo
model = FractalCNN().to(DEVICE)
model.apply(init_weights)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
criterion = nn.CrossEntropyLoss()

# Intentar cargar el modelo guardado si existe y si se especifica la flag
start_epoch = 0
best_val_acc = 0.0
if args.model_preload and os.path.exists('modelos/mejor_modelo.pth'):
    print("Cargando modelo guardado...")
    checkpoint = torch.load('modelos/mejor_modelo.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_acc = checkpoint['val_accuracy']
    print(f"Modelo cargado desde la época {start_epoch} con mejor precisión de validación: {best_val_acc:.2f}%")
else:
    print("Iniciando entrenamiento desde cero.")

def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, num_epochs=30, patience=5, start_epoch=0, best_val_acc=0.0):
    patience_counter = 0
    
    # Crear directorio para guardar el modelo si no existe
    os.makedirs('modelos', exist_ok=True)
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        
        # Evaluación en validación
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
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
        val_accuracy = 100 * correct / total
        
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val Acc: {val_accuracy:.2f}% | Best Val Acc: {best_val_acc:.2f}%")
        
        scheduler.step(val_loss)
        
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            patience_counter = 0
            # Guardar el mejor modelo
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
                'val_loss': val_loss,
            }, 'modelos/mejor_modelo.pth')
            print(f"¡Nuevo mejor modelo guardado con precisión de validación: {val_accuracy:.2f}%!")
        else:
            patience_counter += 1
            if patience_counter >= patience and val_accuracy < 90:
                print("Early stopping triggered.")
                break

def generar_submission():
    # Lista de fractales en orden
    FRACTALES = [
        'tree',
        'koch',
        'dragon',
        'sierpinski_carpet',
        'julia',
        'sierpinski',
        'newton',
        'barnsley',
        'mandelbrot'
    ]

    # Cargar el modelo entrenado
    model = FractalCNN().to(DEVICE)
    checkpoint = torch.load('modelos/mejor_modelo.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Obtener la lista de imágenes en la carpeta test
    test_dir = 'test'  # Asegúrate de que esta es la ruta correcta a tu carpeta de test
    imagenes = [f for f in os.listdir(test_dir) if f.endswith('.png')]

    # Crear el DataFrame con las rutas relativas
    submission_df = pd.DataFrame({
        'relative_path': [f'test/{img}' for img in imagenes]
    })

    # Hacer predicciones
    print("Iniciando predicciones...")
    predictions = []
    for i, img_path in enumerate(submission_df['relative_path']):
        if i % 100 == 0:  # Mostrar progreso cada 100 imágenes
            print(f"Procesando imagen {i}/{len(submission_df)}")
        
        # Cargar y preprocesar la imagen
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)  # Añadir canal
        img = np.expand_dims(img, axis=0)  # Añadir dimensión de batch
        img = torch.tensor(img, dtype=torch.float32).to(DEVICE)
        
        # Hacer la predicción
        model.eval()
        with torch.no_grad():
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)
            pred_label = FRACTALES[predicted.item()]
            predictions.append(pred_label)

    # Actualizar el DataFrame con las predicciones
    submission_df['fractal'] = predictions

    # Guardar el archivo de submission con el nombre específico
    submission_df.to_csv('DragonesYDatos-submission.csv', index=False)
    print("¡Archivo DragonesYDatos-submission.csv generado exitosamente!")

    # Mostrar las primeras filas del archivo de submission
    print("\nPrimeras filas del archivo de submission:")
    print(submission_df.head())

# Primero entrenar el modelo
train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, num_epochs=NUM_EPOCHS, patience=PATIENCE, start_epoch=start_epoch, best_val_acc=best_val_acc)

# Luego generar el archivo de submission
generar_submission()