import os, random, math, yaml, time
import numpy as np
from collections import defaultdict
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import wandb

from torchvision import transforms

# 1) Load config
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

# 2) Initialize W&B
wandb.init(
    project=cfg["project"],
    config=cfg
)
config = wandb.config
config.update({
    "lr":              1e-4,       # Learning rate para mini-ImageNet según HyperMAML
    "weight_decay":    1e-5,
    "embedder_layers": int(config.embedder_layers),
    "hypernet_layers": 3,          # 3 capas como en HyperMAML
    "feature_dim":     int(config.feature_dim),
    "hypernet_hidden": 512,        # 512 unidades como en HyperMAML
    "N_way":           int(config.N_way),
    "K_shot":          int(config.K_shot),
    "Q_query":         int(config.Q_query),
    "lr_milestones":   [1500, 3000], # Para MultiStepLR
    "lr_gamma":        0.3,          # Factor de decay según HyperMAML
    "finetune_embedder": False,      # NO fine-tunear el embedder
}, allow_val_change=True)

# Parámetros adicionales para estabilidad
jitter = 1e-6  # Para estabilidad numérica del kernel
max_grad_norm = 1.0  # Para gradient clipping

# 3) Device & seeds
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(config.seed)
np.random.seed(config.seed)
n_embedder = config.embedder_layers
n_hypernet = config.hypernet_layers
torch.manual_seed(config.seed)

# 4) Data loading (MiniImageNet, on-the-fly)
miniimagenet_root = "/home/aliagatorrens.e/.cache/kagglehub/datasets/arjunashok33/miniimagenet/versions/1"

transform = transforms.Compose([
    transforms.Resize((84,84)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# Build class dictionary: synset -> list of image file paths (not image tensors!)
all_class_dict = defaultdict(list)
for synset in os.listdir(miniimagenet_root):
    class_dir = os.path.join(miniimagenet_root, synset)
    if not os.path.isdir(class_dir): continue
    for img_file in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_file)
        all_class_dict[synset].append(img_path)

def split_classes(class_keys, ratio=0.8):
    uc = list(class_keys); random.shuffle(uc)
    n = int(len(uc)*ratio)
    return uc[:n], uc[n:]

train_cls, test_cls = split_classes(all_class_dict.keys(), 0.8)
meta_train_dict = {c: all_class_dict[c] for c in train_cls}
meta_test_dict  = {c: all_class_dict[c] for c in test_cls}

# 5) Meta‑task sampler (load images on-the-fly)
def sample_meta_task(class_dict, N_way, K_shot, Q_query):
    chosen = random.sample(list(class_dict), N_way)
    sx, sy, qx, qy = [], [], [], []
    for i, c in enumerate(chosen):
        img_paths = class_dict[c]
        idx = random.sample(range(len(img_paths)), K_shot + Q_query)
        # Load images on the fly for support set
        for j in idx[:K_shot]:
            img = Image.open(img_paths[j]).convert("RGB")
            sx.append(transform(img))
            sy.append(i)
        # Load images on the fly for query set
        for j in idx[K_shot:]:
            img = Image.open(img_paths[j]).convert("RGB")
            qx.append(transform(img))
            qy.append(i)
    sx = torch.stack(sx).to(device)
    qx = torch.stack(qx).to(device)
    sy = torch.tensor(sy, device=device).long()
    qy = torch.tensor(qy, device=device).long()
    return (sx, sy), (qx, qy)

# 6) Models
class ConvEmbedder(nn.Module):
    def __init__(self, out_dim, n_layers):
        super().__init__()
        layers, in_ch = [], 3
        ch = out_dim
        for _ in range(n_layers):
            layers += [
                nn.Conv2d(in_ch, ch, 3, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            in_ch = ch
        self.encoder = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1))
        self.out_dim = out_dim

    def forward(self, x):
        h = self.encoder(x)
        return h.view(h.size(0), -1)

class HyperNet(nn.Module):
    def __init__(self, NK, D, H, n_hidden):
        super().__init__()
        # Usar LayerNorm en lugar de BatchNorm para evitar problemas con batch_size=1
        layers = []
        layers += [nn.Linear(NK*(D+config.N_way), H), nn.LayerNorm(H), nn.ReLU(inplace=True)]
        for _ in range(n_hidden-1):
            layers += [nn.Linear(H, H), nn.LayerNorm(H), nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*layers)
        
        # Capas de salida para los hiperparámetros
        self.log_ell = nn.Linear(H, D)
        self.log_sf  = nn.Linear(H, 1)
        self.log_sn  = nn.Linear(H, 1)
        
        # Inicialización cuidadosa
        nn.init.normal_(self.log_ell.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.log_ell.bias, 0.0)  # Inicializa para log_ell ≈ 1.0
        
        nn.init.normal_(self.log_sf.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.log_sf.bias, 0.0)   # Inicializa para log_sf ≈ 1.0
        
        nn.init.normal_(self.log_sn.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.log_sn.bias, -3.0)  # Mantén el valor bajo para sn

    def forward(self, feats, labels_onehot):
        # Normalización L2 para estabilidad numérica
        feats_norm = F.normalize(feats, p=2, dim=1)
        
        # Concatenación y procesamiento
        inp = torch.cat([feats_norm, labels_onehot], dim=1).view(1, -1)
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

# Instanciar embedder y cargar pesos pre-entrenados
print("Loading pretrained embedder...")
embedder = ConvEmbedder(
    out_dim=config.feature_dim,
    n_layers=config.embedder_layers
).to(device)

# Cargar pesos pre-entrenados (si existen)
try:
    ckpt = torch.load("pretrained_conv4d.pth", map_location=device)
    embedder.load_state_dict(ckpt['embedder_state_dict'])
    print("Pretrained embedder loaded successfully.")
except Exception as e:
    print(f"Error loading pretrained embedder: {e}")
    print("Continuing with randomly initialized embedder.")

# CONGELAR EMBEDDER: Cambiar a modo eval y desactivar gradientes
embedder.eval()
for param in embedder.parameters():
    param.requires_grad = False
print("Embedder freezed - will NOT be fine-tuned")

# Monitorear si los parámetros están realmente congelados
trainable_params = [p for p in embedder.parameters() if p.requires_grad]
print(f"Embedder trainable parameters: {len(trainable_params)} (should be 0)")

# Instanciar hypernetwork
hypernet = HyperNet(
    NK=config.N_way*config.K_shot,
    D=config.feature_dim,
    H=config.hypernet_hidden,
    n_hidden=config.hypernet_layers
).to(device)

# Configurar el optimizador - SOLO para la hypernet
param_groups = [
    {'params': hypernet.parameters(), 'lr': config.lr}
]
opt = optim.Adam(param_groups, weight_decay=config.weight_decay)

# Scheduler con MultiStepLR como en HyperMAML
scheduler = MultiStepLR(opt, milestones=config.lr_milestones, gamma=config.lr_gamma)

# Solo monitorizar la hypernet ya que el embedder está congelado
wandb.watch(hypernet, log="all", log_freq=100)

# Monitoreo de parámetros (solo de la hypernetwork)
def monitor_params(ell, sf, sn, hypernet):
    hypernet_grad_norm = 0.0
    for p in hypernet.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            hypernet_grad_norm += param_norm.item() ** 2
    hypernet_grad_norm = hypernet_grad_norm ** 0.5
    
    return {
        "ell_min": ell.min().item(),
        "ell_mean": ell.mean().item(),
        "ell_max": ell.max().item(),
        "sf": sf.item(),
        "sn": sn.item(),
        "hypernet_grad_norm": hypernet_grad_norm
    }

# 7) Meta‑Train Loop
NK = config.N_way * config.K_shot
best_val_acc = 0.0

for ep in tqdm(range(config.META_EPOCHS), desc="Meta‑Epoch"):
    # Solo la hypernet en training mode, el embedder siempre en eval
    hypernet.train()
    
    (sx,sy),(qx,qy) = sample_meta_task(meta_train_dict,
                                       config.N_way,
                                       config.K_shot,
                                       config.Q_query)
    
    # Forward pass por el embedder con torch.no_grad()
    with torch.no_grad():
        z_s = embedder(sx)
        z_q = embedder(qx)
    
    Y_s = F.one_hot(sy, config.N_way).float()

    ell, sf, sn = hypernet(z_s, Y_s)
    K_ss = rbf_kernel(z_s, z_s, ell, sf, jitter)
    K_ss = K_ss + (sn**2) * torch.eye(NK, device=device)
    
    # Resolver el sistema para hacer predicciones con manejo de errores
    try:
        # Intentar usar la descomposición de Cholesky para mayor estabilidad
        L = torch.linalg.cholesky(K_ss)
        temp = torch.triangular_solve(Y_s, L, upper=False)[0]
        alpha = torch.triangular_solve(temp, L.T, upper=True)[0]
    except Exception as e:
        # Fallback a la solución directa si hay problemas numéricos
        print(f"Cholesky failed, using direct solve: {str(e)}")
        alpha = torch.linalg.solve(K_ss, Y_s)

    K_sq = rbf_kernel(z_s, z_q, ell, sf)
    mu_q = K_sq.t() @ alpha

    loss = F.cross_entropy(mu_q, qy)
    acc = (mu_q.argmax(1)==qy).float().mean().item()

    # Backpropagation con gradient clipping (SOLO para hypernet)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(hypernet.parameters(), max_norm=max_grad_norm)
    
    opt.step()
    scheduler.step()

    # Monitoreo de parámetros y logging a W&B cada 10 épocas
    if ep % 10 == 0:
        params_stats = monitor_params(ell, sf, sn, hypernet)
        
        # Log de métricas de entrenamiento
        wandb.log({
            "train/loss": loss.item(),
            "train/acc": acc,
            "params/ell_min": params_stats["ell_min"],
            "params/ell_mean": params_stats["ell_mean"],
            "params/ell_max": params_stats["ell_max"],
            "params/sf": params_stats["sf"],
            "params/sn": params_stats["sn"],
            "grads/hypernet_norm": params_stats["hypernet_grad_norm"],
            "lr": scheduler.get_last_lr()[0]
        }, step=ep)
    else:
        # Log básico para el resto de épocas
        wandb.log({
            "train/loss": loss.item(),
            "train/acc": acc,
        }, step=ep)

    # Validación periódica
    if ep % 50 == 0:
        hypernet.eval()
        val_accs = []
        
        with torch.no_grad():
            for _ in range(20):
                (sx,sy),(qx,qy) = sample_meta_task(meta_test_dict,
                                                  config.N_way,
                                                  config.K_shot,
                                                  config.Q_query)
                z_s = embedder(sx)
                z_q = embedder(qx)
                Y_s = F.one_hot(sy, config.N_way).float()
                
                ell, sf, sn = hypernet(z_s, Y_s)
                K_ss = rbf_kernel(z_s, z_s, ell, sf, jitter)
                K_ss = K_ss + (sn**2) * torch.eye(NK, device=device)
                
                try:
                    L = torch.linalg.cholesky(K_ss)
                    temp = torch.triangular_solve(Y_s, L, upper=False)[0]
                    alpha = torch.triangular_solve(temp, L.T, upper=True)[0]
                except:
                    alpha = torch.linalg.solve(K_ss, Y_s)
                
                mu_q = rbf_kernel(z_s, z_q, ell, sf).t() @ alpha
                preds = mu_q.argmax(1)
                val_accs.append((preds==qy).float().mean().item())
        
        val_acc_mean = np.mean(val_accs)
        wandb.log({"val/acc": val_acc_mean}, step=ep)
        
        tqdm.write(f"Epoch {ep}: Validation Accuracy = {val_acc_mean*100:.2f}%")
        
        # Guardar el mejor modelo
        if val_acc_mean > best_val_acc:
            best_val_acc = val_acc_mean
            torch.save({
                'embedder_state_dict': embedder.state_dict(),
                'hypernet_state_dict': hypernet.state_dict(),
                'epoch': ep,
                'val_acc': val_acc_mean
            }, 'best_hypergp_model.pth')
            tqdm.write(f"  New best model saved! (Acc: {val_acc_mean*100:.2f}%)")
        
        # Volver a modo entrenamiento para hypernet
        hypernet.train()
        # Embedder siempre en eval

# Cargar el mejor modelo para evaluación final
print("\nLoading best model for final evaluation...")
try:
    # Intenta cargar con weights_only=False (más compatible)
    best_ckpt = torch.load('best_hypergp_model.pth', map_location=device, weights_only=False)
except Exception as e:
    print(f"Error loading model with weights_only=False: {str(e)}")
    try:
        # Intenta con la opción predeterminada
        best_ckpt = torch.load('best_hypergp_model.pth', map_location=device)
    except Exception as e:
        print(f"Error loading model with default options: {str(e)}")
        print("Continuing with current model state...")
        best_ckpt = {
            'embedder_state_dict': embedder.state_dict(),
            'hypernet_state_dict': hypernet.state_dict(),
            'epoch': ep,
            'val_acc': best_val_acc
        }

embedder.load_state_dict(best_ckpt['embedder_state_dict'])
hypernet.load_state_dict(best_ckpt['hypernet_state_dict'])

# 8) Test
embedder.eval()
hypernet.eval()
test_accs = []
start = time.time()

with torch.no_grad():
    for _ in tqdm(range(200), desc="Testing"):
        (sx,sy),(qx,qy) = sample_meta_task(meta_test_dict,
                                          config.N_way,
                                          config.K_shot,
                                          config.Q_query)
        z_s = embedder(sx)
        z_q = embedder(qx)
        Y_s = F.one_hot(sy, config.N_way).float()
        
        ell, sf, sn = hypernet(z_s, Y_s)
        K_ss = rbf_kernel(z_s, z_s, ell, sf, jitter) + (sn**2) * torch.eye(NK, device=device)
        
        try:
            L = torch.linalg.cholesky(K_ss)
            temp = torch.triangular_solve(Y_s, L, upper=False)[0]
            alpha = torch.triangular_solve(temp, L.T, upper=True)[0]
        except:
            alpha = torch.linalg.solve(K_ss, Y_s)
        
        mu_q = rbf_kernel(z_s, z_q, ell, sf).t() @ alpha
        preds = mu_q.argmax(1)
        test_accs.append((preds==qy).float().mean().item())

duration = time.time() - start
mean, std = np.mean(test_accs), np.std(test_accs)
ci95 = 1.96 * std / np.sqrt(len(test_accs))

wandb.log({
    "test/acc_mean": mean,
    "test/acc_std": std,
    "test/acc_ci95": ci95,
    "test/time_s": duration
})

print(f"\nTest {config.N_way}‑way/{config.K_shot}‑shot: {mean*100:.2f}% ± {ci95*100:.2f}%")
wandb.finish()