import os, random, time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from torchvision import transforms

# -------------------- Config --------------------
seed             = 42
N_way            = 5
K_shot           = 5
Q_query          = 16
feature_dim      = 64            # Tamaño de embedding según especificaciones
embedder_layers  = 4             # Conv4 backbone (4 capas convolucionales)
hypernet_layers  = 3             # 3 capas como se especifica para CUB
hypernet_hidden  = 512           # 512 unidades como se especifica para CUB
META_EPOCHS      = 4000          # 4000 epochs como se especifica para CUB
device           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuración de fine-tuning
finetune_embedder = True  # Ahora el embedder se fine-tunea

# Learning rates y parámetros de entrenamiento
lr_hypernet      = 1e-3          # 0.001 como se especifica para CUB
lr_embedder      = 1e-4          # Learning rate más bajo para el embedder (10x menor)
weight_decay     = 1e-5          # Mantenemos weight decay para regularización
max_grad_norm    = 1.0           # Para gradient clipping
jitter           = 1e-6          # Para estabilidad numérica del kernel

# Configuración del scheduler según especificaciones
lr_milestones    = [1000, 2000, 3000]  # Puntos donde reducir la learning rate
lr_gamma         = 0.3                # Factor de decaimiento de la learning rate

# Configuración para guardar métricas y resultados
save_dir = "results"
os.makedirs(save_dir, exist_ok=True)
os.makedirs(os.path.join(save_dir, "figures"), exist_ok=True)
os.makedirs(os.path.join(save_dir, "models"), exist_ok=True)
os.makedirs(os.path.join(save_dir, "task_examples"), exist_ok=True)

experiment_name = f"hypergp_{N_way}way_{K_shot}shot_finetune_{finetune_embedder}"
metrics_file = os.path.join(save_dir, f"{experiment_name}_metrics.npz")
model_path = os.path.join(save_dir, "models", f"best_{experiment_name}.pth")

# Para recopilar métricas
train_losses = []
train_accs = []
val_accs = []
val_epochs = []
kernel_params = []  # Lista para almacenar estadísticas de los parámetros del kernel
gradient_norms_hypernet = []  # Para monitorear la estabilidad del entrenamiento de la hypernetwork
gradient_norms_embedder = []  # Para monitorear la estabilidad del entrenamiento del embedder

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# ---------------- Dataset Prep ------------------
dataset_dir = "CUB_200_2011"
images_file = os.path.join(dataset_dir, "images.txt")
labels_file = os.path.join(dataset_dir, "image_class_labels.txt")
image_dir   = os.path.join(dataset_dir, "images")

def load_cub_images():
    imgs, lbs = [], []
    with open(images_file) as f:
        for line in f:
            _, fn = line.strip().split()
            imgs.append(fn)
    with open(labels_file) as f:
        for line in f:
            _, lb = line.strip().split()
            lbs.append(int(lb))
    return imgs, lbs

def build_class_dict(imgs, lbs, transform):
    d = defaultdict(list)
    for fn, lb in zip(imgs, lbs):
        img = Image.open(os.path.join(image_dir, fn)).convert("RGB")
        d[lb].append(transform(img))
    return d

def split_classes(labels, ratio=0.8):
    uc = list(set(labels))
    random.shuffle(uc)
    n = int(len(uc)*ratio)
    return uc[:n], uc[n:]

standard_transform = transforms.Compose([
    transforms.Resize((84,84)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

print("Loading CUB images…")
cub_images, cub_labels = load_cub_images()
train_cls, test_cls   = split_classes(cub_labels)

full_dict      = build_class_dict(cub_images, cub_labels, standard_transform)
meta_train_dict = {c: full_dict[c] for c in train_cls}
meta_test_dict  = {c: full_dict[c] for c in test_cls}
print("Done.")

# --------------- Model Definitions ---------------
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

class HyperNet(nn.Module):
    def __init__(self, NK, D, H, n_hidden):
        super().__init__()
        # Capas de la red - Usando LayerNorm para estabilidad con batch=1
        layers = [nn.Linear(NK*(D+N_way), H), nn.LayerNorm(H), nn.ReLU(inplace=True)]
        for _ in range(n_hidden-1):
            layers += [nn.Linear(H, H), nn.LayerNorm(H), nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*layers)
        
        # Capas de salida para los hiperparámetros
        self.log_ell = nn.Linear(H, D)
        self.log_sf = nn.Linear(H, 1)
        self.log_sn = nn.Linear(H, 1)
        
        # Inicialización cuidadosa
        nn.init.normal_(self.log_ell.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.log_ell.bias, 0.0)  # Inicializa para log_ell ≈ 1.0
        
        nn.init.normal_(self.log_sf.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.log_sf.bias, 0.0)   # Inicializa para log_sf ≈ 1.0
        
        nn.init.normal_(self.log_sn.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.log_sn.bias, -3.0)  # Mantén el valor bajo para sn
        
    def forward(self, feats, labels_onehot):
        # Normalización L2 de características para estabilidad
        feats_norm = F.normalize(feats, p=2, dim=1)
        
        # Concatenación y procesamiento
        inp = torch.cat([feats_norm, labels_onehot], dim=1).view(1,-1)
        h = self.net(inp)
        
        # Generación de hiperparámetros con límites suavizados
        ell = 0.1 + 10.0 * torch.sigmoid(self.log_ell(h))  # [0.1, 10.1]
        sf = 0.5 + 2.0 * torch.sigmoid(self.log_sf(h))     # [0.5, 2.5]
        sn = 1e-3 + 0.1 * torch.sigmoid(self.log_sn(h))    # [0.001, 0.101]
        
        return ell.view(-1), sf.view(-1), sn.view(-1)

def rbf_kernel(X1, X2, ell, sf, jitter=1e-6):
    """
    X1: [n1, D]
    X2: [n2, D]
    ell: [D]        ← one length‐scale per feature dim
    sf: scalar      ← signal variance
    jitter: scalar  ← pequeña constante para estabilidad numérica
    """
    # reshape ell so it broadcasts over the two sample dims:
    ell = ell.view(1, 1, -1)    # [1,1,D]
    diff = (X1.unsqueeze(1) - X2.unsqueeze(0)) / ell  # → [n1,n2,D]
    result = sf**2 * torch.exp(-0.5 * (diff**2).sum(-1))  # [n1,n2]
    
    # Si X1 y X2 son iguales, añadir jitter a la diagonal para estabilidad
    if X1.shape == X2.shape and torch.allclose(X1, X2):
        n = X1.shape[0]
        result = result + jitter * torch.eye(n, device=X1.device)
    
    return result

def sample_meta_task(class_dict):
    # returns (sx,sy),(qx,qy)
    chosen = random.sample(list(class_dict), N_way)
    sx,sy,qx,qy = [],[],[],[]
    for i,c in enumerate(chosen):
        imgs = class_dict[c]
        idxs = random.sample(range(len(imgs)), K_shot+Q_query)
        for j in idxs[:K_shot]:
            sx.append(imgs[j]); sy.append(i)
        for j in idxs[K_shot:]:
            qx.append(imgs[j]); qy.append(i)
    sx = torch.stack(sx).to(device)
    qx = torch.stack(qx).to(device)
    sy = torch.tensor(sy, device=device)
    qy = torch.tensor(qy, device=device)
    return (sx,sy),(qx,qy)

# --------------- Función para visualizar imágenes de ejemplo ---------------
def save_task_example(sx, sy, qx, qy, preds, output_path):
    """Guarda visualizaciones de tareas de ejemplo con predicciones"""
    # Desnormalizar imágenes
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    # Crear visualización
    n_support = sx.size(0)
    n_query = min(10, qx.size(0))  # Mostrar máx. 10 consultas
    
    plt.figure(figsize=(15, 8))
    
    # Mostrar imágenes de soporte
    for i in range(n_support):
        plt.subplot(2, max(n_support, n_query), i+1)
        img = sx[i].cpu()
        img = img * std + mean
        img = img.permute(1, 2, 0).clamp(0, 1).numpy()
        plt.imshow(img)
        plt.title(f"Support: Class {sy[i].item()}")
        plt.axis('off')
    
    # Mostrar imágenes de consulta
    for i in range(n_query):
        plt.subplot(2, max(n_support, n_query), i+1+max(n_support, n_query))
        img = qx[i].cpu()
        img = img * std + mean
        img = img.permute(1, 2, 0).clamp(0, 1).numpy()
        plt.imshow(img)
        correct = (preds[i] == qy[i].item())
        plt.title(f"Query: True {qy[i].item()}, Pred {preds[i]}", 
                  color=('green' if correct else 'red'))
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# --------------- Función para monitorear gradientes e hiperparámetros ---------------
def monitor_params(ep, ell, sf, sn, hypernet, embedder):
    """Monitorea los parámetros del kernel y gradientes"""
    print(f"\nEpoch {ep} monitoring:")
    print(f"  ell range: min={ell.min().item():.4f}, mean={ell.mean().item():.4f}, max={ell.max().item():.4f}")
    print(f"  sf: {sf.item():.4f}, sn: {sn.item():.4f}")
    
    # Norma de gradientes de hypernet
    hypernet_grad_norm = 0.0
    for p in hypernet.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            hypernet_grad_norm += param_norm.item() ** 2
    hypernet_grad_norm = hypernet_grad_norm ** 0.5
    print(f"  HyperNet Gradient norm: {hypernet_grad_norm:.4f}")
    
    # Norma de gradientes del embedder
    embedder_grad_norm = 0.0
    for p in embedder.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            embedder_grad_norm += param_norm.item() ** 2
    embedder_grad_norm = embedder_grad_norm ** 0.5
    print(f"  Embedder Gradient norm: {embedder_grad_norm:.4f}")
    
    # Guardar estadísticas para análisis posterior
    kernel_params.append({
        'epoch': ep,
        'ell_min': ell.min().item(),
        'ell_mean': ell.mean().item(),
        'ell_max': ell.max().item(),
        'sf': sf.item(),
        'sn': sn.item()
    })
    
    gradient_norms_hypernet.append({
        'epoch': ep,
        'norm': hypernet_grad_norm
    })
    
    gradient_norms_embedder.append({
        'epoch': ep,
        'norm': embedder_grad_norm
    })
    
    return hypernet_grad_norm, embedder_grad_norm

# --------------- Load Pretrained Embedder ---------------
print("Loading pretrained embedder...")
ckpt = torch.load("pretrained_conv4d.pth", map_location=device)
embedder = ConvEmbedder(ckpt['feature_dim'], ckpt['n_layers']).to(device)
embedder.load_state_dict(ckpt['embedder_state_dict'])

# Configurar embedder para fine-tuning
if finetune_embedder:
    embedder.train()  # Modo entrenamiento
    print("Embedder configurado para fine-tuning")
else:
    embedder.eval()
    for param in embedder.parameters():
        param.requires_grad = False
    print("Embedder configurado como congelado (no fine-tuning)")

# --------------- Meta-Train HyperNet ---------------
print("Inicializando HyperNet...")
hypernet = HyperNet(N_way*K_shot, feature_dim, hypernet_hidden, hypernet_layers).to(device)

# Configurar el optimizador según si hacemos fine-tuning o no
if finetune_embedder:
    # Dos grupos de parámetros con diferentes LRs
    param_groups = [
        {'params': hypernet.parameters(), 'lr': lr_hypernet},
        {'params': embedder.parameters(), 'lr': lr_embedder}
    ]
    print(f"Optimizando hypernet (LR: {lr_hypernet}) y embedder (LR: {lr_embedder})")
else:
    # Solo optimizar la hypernetwork
    param_groups = [
        {'params': hypernet.parameters(), 'lr': lr_hypernet}
    ]
    print(f"Optimizando solo la hypernetwork (LR: {lr_hypernet})")

optimizer = optim.Adam(param_groups, weight_decay=weight_decay)

# Scheduler según especificaciones
scheduler = MultiStepLR(optimizer, milestones=lr_milestones, gamma=lr_gamma)

# Tiempo de inicio para medir eficiencia
start_time = time.time()

print("Training meta-learner…")
best_val_acc = 0.0
for ep in tqdm(range(1, META_EPOCHS+1)):
    hypernet.train()
    # En cada paso, necesitamos asegurar que el embedder esté en el modo correcto
    if finetune_embedder:
        embedder.train()
    else:
        embedder.eval()
    
    # Muestrear una tarea meta
    (sx,sy),(qx,qy) = sample_meta_task(meta_train_dict)
    
    # Obtener embeddings (con o sin gradientes según la configuración)
    if finetune_embedder:
        z_s = embedder(sx)   # [NK, D]
        z_q = embedder(qx)   # [Q_query*N_way, D]
    else:
        with torch.no_grad():
            z_s = embedder(sx)   # [NK, D]
            z_q = embedder(qx)   # [Q_query*N_way, D]
    
    # One-hot encoding de las etiquetas
    Y_s = F.one_hot(sy, N_way).float()
    
    # Obtener hiperparámetros del kernel para este episodio
    ell, sf, sn = hypernet(z_s, Y_s)
    
    # Construir matrices de kernel
    K_ss = rbf_kernel(z_s, z_s, ell, sf) + (sn**2) * torch.eye(N_way*K_shot, device=device)
    K_sq = rbf_kernel(z_s, z_q, ell, sf)
    
    # Resolver el sistema para hacer predicciones
    try:
        # Intentar usar la descomposición de Cholesky para mayor estabilidad
        L = torch.linalg.cholesky(K_ss)
        temp = torch.triangular_solve(Y_s, L, upper=False)[0]
        alpha = torch.triangular_solve(temp, L.T, upper=True)[0]
    except Exception as e:
        # Fallback a la solución directa si hay problemas numéricos
        print(f"Cholesky failed, using direct solve: {str(e)}")
        alpha = torch.linalg.solve(K_ss, Y_s)
    
    # Calcular predicciones
    mu_q = K_sq.T @ alpha
    
    # Calcular pérdida
    loss = F.cross_entropy(mu_q, qy)
    
    # Calcular accuracy
    acc = (mu_q.argmax(1) == qy).float().mean().item()
    
    # Registrar métricas
    train_losses.append(loss.item())
    train_accs.append(acc)
    
    # Backpropagation con gradient clipping
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(hypernet.parameters(), max_norm=max_grad_norm)
    if finetune_embedder:
        torch.nn.utils.clip_grad_norm_(embedder.parameters(), max_norm=max_grad_norm)
    
    optimizer.step()
    scheduler.step()
    
    # Monitoreo periódico de parámetros
    if ep % 100 == 0:
        hypernet_grad_norm, embedder_grad_norm = monitor_params(ep, ell, sf, sn, hypernet, embedder)
        print(f"[Episode {ep}/{META_EPOCHS}] Loss: {loss.item():.4f}, Acc: {acc*100:.2f}%")
        print(f"Grad norms - HyperNet: {hypernet_grad_norm:.4f}, Embedder: {embedder_grad_norm:.4f}")
    
    # Validación periódica
    if ep % 200 == 0:
        hypernet.eval()
        embedder.eval()  # Siempre eval para validación
        val_accs_this_epoch = []
        
        print(f"\nValidating at epoch {ep}...")
        with torch.no_grad():
            for task_idx in range(50):  # Evaluar en 50 tareas
                (sx,sy),(qx,qy) = sample_meta_task(meta_test_dict)
                z_s = embedder(sx)
                z_q = embedder(qx)
                Y_s = F.one_hot(sy, N_way).float()
                
                ell, sf, sn = hypernet(z_s, Y_s)
                K_ss = rbf_kernel(z_s, z_s, ell, sf) + (sn**2) * torch.eye(N_way*K_shot, device=device)
                K_sq = rbf_kernel(z_s, z_q, ell, sf)
                
                try:
                    L = torch.linalg.cholesky(K_ss)
                    temp = torch.triangular_solve(Y_s, L, upper=False)[0]
                    alpha = torch.triangular_solve(temp, L.T, upper=True)[0]
                except:
                    alpha = torch.linalg.solve(K_ss, Y_s)
                
                mu_q = K_sq.T @ alpha
                preds = mu_q.argmax(1).cpu()
                task_acc = (preds == qy.cpu()).float().mean().item()
                val_accs_this_epoch.append(task_acc)
                
                # Guardar visualización para la primera tarea
                if task_idx == 0 and (ep == 200 or ep == META_EPOCHS):
                    example_path = os.path.join(save_dir, "task_examples", 
                                               f"task_example_epoch_{ep}.png")
                    save_task_example(sx, sy, qx, qy, preds, example_path)
        
        # Calcular estadísticas de validación
        val_acc_mean = np.mean(val_accs_this_epoch)
        val_acc_std = np.std(val_accs_this_epoch)
        
        # Registrar métricas de validación
        val_accs.append(val_acc_mean)
        val_epochs.append(ep)
        
        print(f"Validation at epoch {ep}: Acc = {val_acc_mean*100:.2f}% ± {val_acc_std*100:.2f}%")
        
        # Guardar el mejor modelo
        if val_acc_mean > best_val_acc:
            best_val_acc = val_acc_mean
            torch.save({
                'embedder_state_dict': embedder.state_dict(),
                'hypernet_state_dict': hypernet.state_dict(),
                'epoch': ep,
                'val_acc': val_acc_mean,
                'val_acc_std': val_acc_std,
                'n_way': N_way,
                'k_shot': K_shot,
                'finetune_embedder': finetune_embedder,
                'kernel_params': kernel_params[-1] if kernel_params else None
            }, model_path)
            print(f"  New best model saved! (Acc: {val_acc_mean*100:.2f}%)")
        
        # Regresar a modo entrenamiento si es necesario
        hypernet.train()
        if finetune_embedder:
            embedder.train()

# Tiempo total de entrenamiento
training_time = time.time() - start_time
print(f"\nTraining completed in {training_time:.2f} seconds")

# Guardar métricas para análisis posterior
np.savez(metrics_file,
         train_losses=train_losses,
         train_accs=train_accs,
         val_accs=val_accs,
         val_epochs=val_epochs,
         kernel_params=kernel_params,
         gradient_norms_hypernet=gradient_norms_hypernet,
         gradient_norms_embedder=gradient_norms_embedder,
         training_time=training_time)

# --------------- Visualización de Curvas de Aprendizaje ---------------
print("Generating learning curves...")
plt.figure(figsize=(15, 5))

# Curva de pérdida de entrenamiento
plt.subplot(1, 2, 1)
plt.plot(np.arange(1, len(train_losses)+1), train_losses)
plt.title('Training Loss')
plt.xlabel('Episodes')
plt.ylabel('Loss')
plt.grid(True)

# Curva de precisión de validación
plt.subplot(1, 2, 2)
plt.plot(val_epochs, val_accs, marker='o')
plt.title('Validation Accuracy')
plt.xlabel('Episodes')
plt.ylabel('Accuracy')
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "figures", f"{experiment_name}_learning_curves.png"))
plt.close()

# --------------- Visualización de Parámetros del Kernel ---------------
print("Generating kernel parameter visualizations...")
if kernel_params:
    # Extraer datos
    epochs = [item['epoch'] for item in kernel_params]
    ell_mins = [item['ell_min'] for item in kernel_params]
    ell_means = [item['ell_mean'] for item in kernel_params]
    ell_maxs = [item['ell_max'] for item in kernel_params]
    sfs = [item['sf'] for item in kernel_params]
    sns = [item['sn'] for item in kernel_params]
    
    plt.figure(figsize=(15, 10))
    
    # Evolución de ell (lengthscales)
    plt.subplot(2, 2, 1)
    plt.plot(epochs, ell_mins, label='Min')
    plt.plot(epochs, ell_means, label='Mean')
    plt.plot(epochs, ell_maxs, label='Max')
    plt.title('Lengthscale (ell) Evolution')
    plt.xlabel('Episodes')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    # Evolución de sf (signal variance)
    plt.subplot(2, 2, 2)
    plt.plot(epochs, sfs)
    plt.title('Signal Amplitude (sf) Evolution')
    plt.xlabel('Episodes')
    plt.ylabel('Value')
    plt.grid(True)
    
    # Evolución de sn (noise level)
    plt.subplot(2, 2, 3)
    plt.plot(epochs, sns)
    plt.title('Noise Level (sn) Evolution')
    plt.xlabel('Episodes')
    plt.ylabel('Value')
    plt.grid(True)
    
    # Evolución de gradientes
    plt.subplot(2, 2, 4)
    if gradient_norms_hypernet:
        hypernet_epochs = [item['epoch'] for item in gradient_norms_hypernet]
        hypernet_values = [item['norm'] for item in gradient_norms_hypernet]
        plt.plot(hypernet_epochs, hypernet_values, label='HyperNet')
    
    if finetune_embedder and gradient_norms_embedder:
        embedder_epochs = [item['epoch'] for item in gradient_norms_embedder]
        embedder_values = [item['norm'] for item in gradient_norms_embedder]
        plt.plot(embedder_epochs, embedder_values, label='Embedder')
    
    plt.title('Gradient Norm Evolution')
    plt.xlabel('Episodes')
    plt.ylabel('Norm')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "figures", f"{experiment_name}_kernel_params.png"))
    plt.close()

# --------------- Cargar el mejor modelo para evaluación final ---------------
print("\nLoading best model for final evaluation...")
try:
    # Intentar cargar con weights_only=False (más seguro para versiones recientes de PyTorch)
    best_ckpt = torch.load(model_path, map_location=device, weights_only=False)
except Exception as e:
    print(f"Error loading with weights_only=False: {e}")
    try:
        # Intentar carga básica
        best_ckpt = torch.load(model_path, map_location=device)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Continuing with current state...")
        best_ckpt = {
            'embedder_state_dict': embedder.state_dict(),
            'hypernet_state_dict': hypernet.state_dict(),
            'epoch': META_EPOCHS,
            'val_acc': best_val_acc
        }

embedder.load_state_dict(best_ckpt['embedder_state_dict'])
hypernet.load_state_dict(best_ckpt['hypernet_state_dict'])
best_epoch = best_ckpt.get('epoch', 'unknown')
print(f"Model from epoch {best_epoch} loaded with validation accuracy: {best_ckpt.get('val_acc', 0.0)*100:.2f}%")

# --------------- Evaluación Final -------------------
print("Evaluating…")
embedder.eval()
hypernet.eval()
test_accs = []
task_confidences = []  # Para análisis de calibración
test_times = []        # Para medir rendimiento

all_task_stats = []    # Para análisis detallado por tarea
calibration_bins = 10  # Número de bins para diagnóstico de calibración

with torch.no_grad():
    for task_idx in tqdm(range(600)):
        task_start = time.time()
        
        (sx,sy),(qx,qy) = sample_meta_task(meta_test_dict)
        z_s = embedder(sx)
        z_q = embedder(qx)
        Y_s = F.one_hot(sy, N_way).float()
        
        ell, sf, sn = hypernet(z_s, Y_s)
        K_ss = rbf_kernel(z_s, z_s, ell, sf) + (sn**2) * torch.eye(N_way*K_shot, device=device)
        K_sq = rbf_kernel(z_s, z_q, ell, sf)
        
        try:
            L = torch.linalg.cholesky(K_ss)
            temp = torch.triangular_solve(Y_s, L, upper=False)[0]
            alpha = torch.triangular_solve(temp, L.T, upper=True)[0]
        except:
            alpha = torch.linalg.solve(K_ss, Y_s)
        
        mu_q = K_sq.T @ alpha
        
        # Aplicar softmax para obtener probabilidades
        probs = F.softmax(mu_q, dim=1)
        
        # Calcular predicciones y confidencias
        preds = mu_q.argmax(1).cpu()
        confidences = probs.max(1)[0].cpu()
        
        # Calcular accuracy
        accuracy = (preds == qy.cpu()).float().mean().item()
        test_accs.append(accuracy)
        
        # Tiempo por tarea
        task_time = time.time() - task_start
        test_times.append(task_time)
        
        # Guardar estadísticas para análisis de calibración
        task_confidences.append(confidences.numpy())
        
        # Guardar estadísticas detalladas de la tarea
        all_task_stats.append({
            'task_id': task_idx,
            'accuracy': accuracy,
            'confidences': confidences.numpy(),
            'ell_stats': {
                'min': ell.min().item(),
                'mean': ell.mean().item(),
                'max': ell.max().item()
            },
            'sf': sf.item(),
            'sn': sn.item(),
            'time': task_time
        })
        
        # Guardar ejemplos de tareas específicas (algunas exitosas, algunas fallidas)
        if task_idx in [0, 100, 200]:
            example_path = os.path.join(save_dir, "task_examples", 
                                       f"test_task_{task_idx}_acc_{accuracy:.2f}.png")
            save_task_example(sx, sy, qx, qy, preds, example_path)

# Calcular estadísticas del test
mean_acc = np.mean(test_accs)
std_acc = np.std(test_accs)
ci95 = 1.96 * std_acc / np.sqrt(len(test_accs))
median_acc = np.median(test_accs)
min_acc = np.min(test_accs)
max_acc = np.max(test_accs)

# Estadísticas de tiempo
mean_time = np.mean(test_times)
total_time = np.sum(test_times)

# Imprimir resultados
print(f"\nTest Results for {N_way}-way {K_shot}-shot:")
print(f"  Accuracy: {mean_acc*100:.2f}% ± {ci95*100:.2f}%")
print(f"  Standard deviation: {std_acc*100:.2f}%")
print(f"  Median: {median_acc*100:.2f}%, Min: {min_acc*100:.2f}%, Max: {max_acc*100:.2f}%")
print(f"  Time per task: {mean_time*1000:.2f} ms, Total time: {total_time:.2f} s")

# Guardar resultados en archivo
results = {
    'mean_acc': mean_acc,
    'std_acc': std_acc,
    'ci95': ci95,
    'median_acc': median_acc,
    'min_acc': min_acc,
    'max_acc': max_acc,
    'mean_time': mean_time,
    'total_time': total_time,
    'test_accs': test_accs,
    'task_stats': all_task_stats
}
np.save(os.path.join(save_dir, f"{experiment_name}_test_results.npy"), results)

# --------------- Visualizaciones Adicionales ---------------
# 1. Histograma de Precisión por Tarea
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(test_accs, bins=20, alpha=0.7, color='#1f77b4', edgecolor='black')
plt.axvline(mean_acc, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_acc*100:.2f}%')
plt.axvline(median_acc, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_acc*100:.2f}%')
plt.title(f'Accuracy Distribution ({N_way}-way {K_shot}-shot)')
plt.xlabel('Accuracy')
plt.ylabel('Number of Tasks')
plt.legend()

# 2. Boxplot de Precisión
plt.subplot(1, 2, 2)
plt.boxplot(test_accs, vert=True, patch_artist=True)
plt.title('Accuracy Boxplot')
plt.ylabel('Accuracy')
plt.grid(True, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "figures", f"{experiment_name}_accuracy_distribution.png"))
plt.close()

# 3. Distribución de Parámetros del Kernel en las Tareas de Test
ell_mins = [stats['ell_stats']['min'] for stats in all_task_stats]
ell_means = [stats['ell_stats']['mean'] for stats in all_task_stats]
ell_maxs = [stats['ell_stats']['max'] for stats in all_task_stats]
sfs = [stats['sf'] for stats in all_task_stats]
sns = [stats['sn'] for stats in all_task_stats]

plt.figure(figsize=(15, 10))

# Lengthscales
plt.subplot(2, 2, 1)
plt.hist(ell_means, bins=20, alpha=0.7, color='#1f77b4', edgecolor='black')
plt.title('Lengthscale (ell) Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Signal variance
plt.subplot(2, 2, 2)
plt.hist(sfs, bins=20, alpha=0.7, color='#ff7f0e', edgecolor='black')
plt.title('Signal Amplitude (sf) Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Noise level
plt.subplot(2, 2, 3)
plt.hist(sns, bins=20, alpha=0.7, color='#2ca02c', edgecolor='black')
plt.title('Noise Level (sn) Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Relación entre parámetros y precisión
plt.subplot(2, 2, 4)
plt.scatter(ell_means, test_accs, alpha=0.5, color='#1f77b4', edgecolor='black')
plt.title('Lengthscale vs. Accuracy')
plt.xlabel('Mean Lengthscale (ell)')
plt.ylabel('Accuracy')
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "figures", f"{experiment_name}_kernel_params_distribution.png"))
plt.close()

# 4. Análisis de Calibración
# Convertir listas de confidencias a un array único
all_confidences = np.concatenate([conf for conf in task_confidences])
all_correctness = np.concatenate([(preds == qy.cpu()).float().numpy() for preds, qy in 
                                 zip([a.argmax(1).cpu() for a in [mu_q] * 600], 
                                     [qy.cpu() for (_, _), (_, qy) in 
                                      [sample_meta_task(meta_test_dict) for _ in range(600)]])])

# Crear bins de confidencia
bin_edges = np.linspace(0, 1, calibration_bins + 1)
bin_indices = np.digitize(all_confidences, bin_edges[1:-1])

# Calcular precisión y confianza promedio por bin
bin_accs = np.zeros(calibration_bins)
bin_confs = np.zeros(calibration_bins)
bin_counts = np.zeros(calibration_bins)

for i in range(len(all_confidences)):
    bin_idx = bin_indices[i]
    bin_accs[bin_idx] += all_correctness[i]
    bin_confs[bin_idx] += all_confidences[i]
    bin_counts[bin_idx] += 1

# Calcular promedios (evitar división por cero)
for i in range(calibration_bins):
    if bin_counts[i] > 0:
        bin_accs[i] /= bin_counts[i]
        bin_confs[i] /= bin_counts[i]

# Calcular Expected Calibration Error (ECE)
ece = np.sum(np.abs(bin_accs - bin_confs) * (bin_counts / len(all_confidences)))

# Diagrama de fiabilidad (Calibration Plot)
plt.figure(figsize=(10, 10))
plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
plt.bar(bin_edges[:-1], bin_accs, width=1/calibration_bins, alpha=0.8, align='edge', 
        edgecolor='black', color='#1f77b4', label='Accuracy in bin')
plt.plot(bin_confs, bin_accs, 'ro-', label='Accuracy vs Confidence')

plt.title(f'Calibration Plot (ECE: {ece:.4f})')
plt.xlabel('Confidence')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, "figures", f"{experiment_name}_calibration.png"))
plt.close()

# 5. Comparación de Gradientes entre HyperNet y Embedder para Fine-tuning
if finetune_embedder and gradient_norms_hypernet and gradient_norms_embedder:
    plt.figure(figsize=(10, 6))
    
    # Extraer datos
    hypernet_epochs = [item['epoch'] for item in gradient_norms_hypernet]
    hypernet_values = [item['norm'] for item in gradient_norms_hypernet]
    
    embedder_epochs = [item['epoch'] for item in gradient_norms_embedder]
    embedder_values = [item['norm'] for item in gradient_norms_embedder]
    
    plt.plot(hypernet_epochs, hypernet_values, label='HyperNet')
    plt.plot(embedder_epochs, embedder_values, label='Embedder')
    
    plt.title('Gradient Norm Comparison')
    plt.xlabel('Episodes')
    plt.ylabel('Gradient Norm')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(os.path.join(save_dir, "figures", f"{experiment_name}_gradient_comparison.png"))
    plt.close()

# 6. Tabla de Comparación
# Crear tabla comparativa con otros métodos
# (Esto requiere datos externos, el código es solo un ejemplo de formato)

comparison_methods = {
    f"HyperGP (Finetune={finetune_embedder})": {
        "accuracy": mean_acc * 100,
        "ci": ci95 * 100
    },
    "HyperGP (Frozen Embedder)": {
        "accuracy": 45.5,  # Valores de ejemplo - actualizar con resultados reales
        "ci": 2.1
    },
    "ProtoNet": {
        "accuracy": 50.2,  # Valores de ejemplo - actualizar con resultados reales
        "ci": 1.8
    }
}

# Formatear tabla como texto
print("\nComparison Table:")
print("-" * 60)
print(f"{'Method':<30} {'Accuracy':<10} {'95% CI':<10}")
print("-" * 60)
for method, results in comparison_methods.items():
    print(f"{method:<30} {results['accuracy']:.2f}% ±{results['ci']:.2f}%")
print("-" * 60)

# 7. Resumen Final
print("\nExperiment Summary:")
print(f"Model: HyperGP with {hypernet_layers} hypernet layers ({hypernet_hidden} units)")
print(f"Dataset: CUB-200-2011, {N_way}-way {K_shot}-shot")
print(f"Embedder: {'Fine-tuned' if finetune_embedder else 'Frozen'} Conv4 network")
print(f"Best validation accuracy: {best_val_acc*100:.2f}% at epoch {best_epoch}")
print(f"Test accuracy: {mean_acc*100:.2f}% ± {ci95*100:.2f}%")
print(f"Training time: {training_time:.2f} seconds")
print(f"Average inference time per task: {mean_time*1000:.2f} ms")
print("\nAll results and visualizations saved to:", save_dir)