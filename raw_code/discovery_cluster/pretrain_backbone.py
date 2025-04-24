import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Configuración
seed = 42
batch_size = 128
PRETRAIN_EPOCHS = 400
RESUME_FROM_EPOCH = 250  # Start from this checkpoint
START_EPOCH = RESUME_FROM_EPOCH  # For loop control
learning_rate = 0.001
feature_dim = 64
embedder_layers = 4
weight_decay = 5e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

# Configuración de rutas
miniimagenet_root = "/home/aliagatorrens.e/.cache/kagglehub/datasets/arjunashok33/miniimagenet/versions/1"

# Fijamos las semillas para reproducibilidad
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Transformaciones para las imágenes
train_transform = transforms.Compose([
    transforms.Resize((84, 84)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Modelo del embedder
class ConvEmbedder(nn.Module):
    def __init__(self, out_dim, n_layers):
        super().__init__()
        layers, in_ch = [], 3
        for _ in range(n_layers):
            layers += [
                nn.Conv2d(in_ch, out_dim, 3, padding=1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            in_ch = out_dim
        self.encoder = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1))
    
    def forward(self, x):
        return self.encoder(x).view(x.size(0), -1)

# Clase para el clasificador de cabeza
class Classifier(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.classifier = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.classifier(x)

# Dataset personalizado para MiniImageNet
class MiniImageNetDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []
        self.targets = []
        self.classes = []
        
        # Usar todas las clases disponibles
        self.synsets = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        
        # Crear mapeo de etiquetas
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.synsets)}
        
        # Recopilar todas las muestras
        for class_name in self.synsets:
            class_dir = os.path.join(root, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            self.classes.append(class_name)
            class_idx = self.class_to_idx[class_name]
            
            for img_file in os.listdir(class_dir):
                if not img_file.lower().endswith(('png', 'jpg', 'jpeg')):
                    continue
                    
                img_path = os.path.join(class_dir, img_file)
                self.samples.append((img_path, class_idx))
                self.targets.append(class_idx)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, target = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target

# Crear dataset y dividirlo en entrenamiento/validación
full_dataset = MiniImageNetDataset(miniimagenet_root, transform=train_transform)
num_samples = len(full_dataset)
num_train = int(0.9 * num_samples)
num_val = num_samples - num_train

train_dataset, val_dataset = random_split(
    full_dataset, 
    [num_train, num_val],
    generator=torch.Generator().manual_seed(seed)
)

# Crear dataloaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=4, 
    pin_memory=True
)

# Crear modelos
embedder = ConvEmbedder(feature_dim, embedder_layers).to(device)
classifier = Classifier(feature_dim, len(full_dataset.classes)).to(device)

# Optimizador y criterio de pérdida
optimizer = optim.Adam(
    list(embedder.parameters()) + list(classifier.parameters()),
    lr=learning_rate, 
    weight_decay=weight_decay
)
criterion = nn.CrossEntropyLoss()

# Cargar checkpoint de la época 250
checkpoint_path = f"embedder_epoch_{RESUME_FROM_EPOCH}.pth"
if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from epoch {RESUME_FROM_EPOCH}...")
    try:
        # Intenta cargar con weights_only=False (enfoque más seguro para checkpoints propios)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Standard loading failed: {e}")
        try:
            # Intenta con safe_globals (enfoque alternativo)
            from torch.serialization import add_safe_globals
            add_safe_globals(['scalar'], numpy_module=np)
            checkpoint = torch.load(checkpoint_path, map_location=device)
        except Exception as e:
            print(f"Safe globals loading failed: {e}")
            print(f"Checkpoint file {checkpoint_path} could not be loaded. Cannot resume training.")
            exit(1)
            
    embedder.load_state_dict(checkpoint['embedder_state_dict'])
    classifier.load_state_dict(checkpoint['classifier_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Successfully loaded checkpoint. Resuming training from epoch {RESUME_FROM_EPOCH}...")
else:
    print(f"Checkpoint file {checkpoint_path} not found. Cannot resume training.")
    exit(1)

# Pre-entrenamiento del embedder (continuando desde la época 250)
print(f"Resuming training from epoch {START_EPOCH} to {PRETRAIN_EPOCHS}...")
for epoch in tqdm(range(START_EPOCH, PRETRAIN_EPOCHS)):
    losses, accs = [], []
    embedder.train()
    classifier.train()

    for x, y in train_loader:
        x, y = x.to(device), y.to(device).long()

        # Forward pass
        z = embedder(x)
        logits = classifier(z)

        # Calcular pérdida y aplicar backprop
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Registrar estadísticas
        losses.append(loss.item())
        accs.append((logits.argmax(1) == y).float().mean().item())

    # Calcular promedios para la época
    epoch_loss = np.mean(losses)
    epoch_acc = np.mean(accs)

    # Mostrar progreso cada 50 epochs
    if (epoch+1) % 50 == 0 or epoch == START_EPOCH:
        print(f"Epoch {epoch+1}/{PRETRAIN_EPOCHS} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc*100:.2f}%")
        # Guardar checkpoint cada 50 épocas
        checkpoint_path = f"embedder_epoch_{epoch+1}.pth"
        torch.save({
            'n_layers': embedder_layers,
            'feature_dim': feature_dim,
            'embedder_state_dict': embedder.state_dict(),
            'classifier_state_dict': classifier.state_dict(),
            'epoch': epoch + 1,
            'train_acc': epoch_acc,
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path)
        print(f"  Checkpoint saved to {checkpoint_path}")

# Guardar el modelo pre-entrenado final
torch.save({
    'n_layers': embedder_layers,
    'feature_dim': feature_dim,
    'embedder_state_dict': embedder.state_dict()
}, "pretrained_conv4d_mini.pth")

print("Saved pretrained model to pretrained_conv4d_mini.pth")