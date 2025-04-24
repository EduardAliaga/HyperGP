import os, random
import numpy as np
from collections import defaultdict
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from torchvision import transforms

# -------------------- Config --------------------
seed             = 42
N_way            = 5
K_shot           = 5
Q_query          = 16
feature_dim      = 64
embedder_layers  = 4
hypernet_layers  = 3
hypernet_hidden  = 128
META_EPOCHS      = 600
device           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Learning rates y parámetros de entrenamiento
lr_hypernet      = 1e-4
lr_embedder      = 5e-6  # Tasa aún más baja para el embedder para mayor estabilidad
max_grad_norm    = 1.0   # Para gradient clipping
jitter           = 1e-6  # Para estabilidad numérica del kernel

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
        # Capas de la red - usando LayerNorm en lugar de BatchNorm1d
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

# --------------- Función para monitorear gradientes e hiperparámetros ---------------
def monitor_params(ep, ell, sf, sn, hypernet, embedder):
    print(f"\nEpoch {ep} monitoring:")
    print(f"  ell range: min={ell.min().item():.4f}, mean={ell.mean().item():.4f}, max={ell.max().item():.4f}")
    print(f"  sf: {sf.item():.4f}, sn: {sn.item():.4f}")
    
    # Norma de gradientes de la hypernetwork
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

# --------------- Load Pretrained Embedder ---------------
ckpt = torch.load("pretrained_conv4d.pth", map_location=device)
embedder = ConvEmbedder(ckpt['feature_dim'], ckpt['n_layers']).to(device)
embedder.load_state_dict(ckpt['embedder_state_dict'])
embedder.train()  # Poner en modo entrenamiento desde el inicio

# --------------- Meta-Train HyperNet ---------------
hypernet = HyperNet(N_way*K_shot, feature_dim, hypernet_hidden, hypernet_layers).to(device)

# Configurar el optimizador con ambas redes desde el inicio
param_groups = [
    {'params': hypernet.parameters(), 'lr': lr_hypernet},
    {'params': embedder.parameters(), 'lr': lr_embedder}  # Usar una tasa más baja para el embedder
]
optimizer = optim.Adam(param_groups, weight_decay=1e-5)

# Scheduler con warmup
def warmup_lambda(epoch):
    if epoch < 10:
        return 0.1 + 0.9 * (epoch / 10)  # 10% a 100% en 10 epochs
    return 1.0

scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)

print("Training meta-learner…")
best_val_acc = 0.0
for ep in tqdm(range(1, META_EPOCHS+1)):
    hypernet.train()
    embedder.train()  # Asegurar que ambas redes estén en modo entrenamiento
    
    # Muestrear una tarea meta
    (sx,sy),(qx,qy) = sample_meta_task(meta_train_dict)
    
    # Obtener embeddings
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
    
    # Backpropagation con gradient clipping
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(hypernet.parameters(), max_norm=max_grad_norm)
    torch.nn.utils.clip_grad_norm_(embedder.parameters(), max_norm=max_grad_norm)
    
    optimizer.step()
    scheduler.step()
    
    # Monitoreo periódico
    if ep % 50 == 0:
        monitor_params(ep, ell, sf, sn, hypernet, embedder)
    
    # Informar progreso y evaluar en validación
    if ep % 100 == 0:
        print(f"[Episode {ep}/{META_EPOCHS}] Loss: {loss.item():.4f}")
        
        # Evaluación rápida en un subconjunto de tareas de validación
        embedder.eval()
        hypernet.eval()
        val_accs = []
        
        with torch.no_grad():
            for _ in range(50):  # Evaluar en 50 tareas
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
                val_accs.append((mu_q.argmax(1)==qy).float().mean().item())
        
        val_acc = np.mean(val_accs)
        print(f"  Validation Accuracy: {val_acc*100:.2f}%")
        
        # Guardar el mejor modelo
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'embedder_state_dict': embedder.state_dict(),
                'hypernet_state_dict': hypernet.state_dict(),
                'epoch': ep,
                'val_acc': val_acc
            }, 'best_hypergp_model.pth')
            print(f"  New best model saved! (Acc: {val_acc*100:.2f}%)")
        
        # Volver a modo entrenamiento
        embedder.train()
        hypernet.train()

# --------------- Cargar el mejor modelo para evaluación final ---------------
print("\nLoading best model for final evaluation...")
best_ckpt = torch.load('best_hypergp_model.pth', map_location=device)
embedder.load_state_dict(best_ckpt['embedder_state_dict'])
hypernet.load_state_dict(best_ckpt['hypernet_state_dict'])

# --------------- Evaluación Final -------------------
print("Evaluating…")
embedder.eval()
hypernet.eval()
accs = []

with torch.no_grad():
    for _ in tqdm(range(600)):
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
        accs.append((mu_q.argmax(1)==qy).float().mean().item())

mean, std = np.mean(accs), np.std(accs)
print(f"Test Accuracy: {mean*100:.2f}% ± {1.96*std/np.sqrt(len(accs))*100:.2f}%")