import os, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from collections import defaultdict
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

#— Device & Seeds
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42); np.random.seed(42); random.seed(42)

#— Hyper‑params
SUPER_TUNE_EPOCHS = 5    # epochs to train ConvEmbedder on base classes
META_FINE_TUNE     = True # True → allow ConvEmbedder last conv‑block update in meta‑train
N_way, K_shot, Q_query = 5, 5, 15

############################################
# 1. Load CUB & split classes
############################################
dataset_dir = "CUB_200_2011"
images_file = os.path.join(dataset_dir, "images.txt")
labels_file = os.path.join(dataset_dir, "image_class_labels.txt")
image_dir   = os.path.join(dataset_dir, "images")

transform = transforms.Compose([
    transforms.Resize((84,84)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225]),
])

cub_images, cub_labels = [], []
with open(images_file) as f:
    for line in f:
        _, fn = line.strip().split()
        cub_images.append(fn)
with open(labels_file) as f:
    for line in f:
        _, lb = line.strip().split()
        cub_labels.append(int(lb))

all_class_dict = defaultdict(list)
for fn, lb in zip(cub_images, cub_labels):
    img = Image.open(os.path.join(image_dir, fn)).convert("RGB")
    all_class_dict[lb].append(transform(img))

def split_classes(labels, ratio=0.8):
    uc = list(set(labels)); random.shuffle(uc)
    n  = int(len(uc)*ratio)
    return uc[:n], uc[n:]

train_cls, test_cls = split_classes(cub_labels, 0.8)
meta_train_dict = {c: all_class_dict[c] for c in train_cls}
meta_test_dict  = {c: all_class_dict[c] for c in test_cls}

############################################
# 2. Meta‑Task Sampler
############################################
def sample_meta_task_from_dict(class_dict, N_way=5, K_shot=5, Q_query=15):
    chosen = random.sample(list(class_dict), N_way)
    sx, sy, qx, qy = [], [], [], []
    for i, c in enumerate(chosen):
        imgs = class_dict[c]
        idx  = random.sample(range(len(imgs)), K_shot + Q_query)
        for j in idx[:K_shot]:
            sx.append(imgs[j]); sy.append(i)
        for j in idx[K_shot:]:
            qx.append(imgs[j]); qy.append(i)
    sx = torch.stack(sx).to(device)
    qx = torch.stack(qx).to(device)
    sy = torch.tensor(sy,device=device).long().unsqueeze(1)
    qy = torch.tensor(qy,device=device).long().unsqueeze(1)
    return (sx, sy), (qx, qy)

############################################
# 3. Supervised pre‑train on base classes
############################################
# flatten base classes
base_classes = sorted(meta_train_dict.keys())
num_base     = len(base_classes)
base_data    = [(img, new_lbl)
                for new_lbl, cls in enumerate(base_classes)
                for img in meta_train_dict[cls]]

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        img, lbl = self.data[idx]
        return img, lbl

base_loader = torch.utils.data.DataLoader(
    BaseDataset(base_data),
    batch_size=64, shuffle=True, num_workers=0, pin_memory=True
)

# 3.1 Your 4‑layer ConvEmbedder
class ConvEmbedder(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3,64,3,padding=1), nn.ReLU(),
            nn.BatchNorm2d(64), nn.MaxPool2d(2),

            nn.Conv2d(64,64,3,padding=1), nn.ReLU(),
            nn.BatchNorm2d(64), nn.MaxPool2d(2),

            nn.Conv2d(64,64,3,padding=1), nn.ReLU(),
            nn.BatchNorm2d(64), nn.MaxPool2d(2),

            nn.Conv2d(64,64,3,padding=1), nn.ReLU(),
            nn.BatchNorm2d(64), nn.AdaptiveAvgPool2d(1),
        )
        self.out_dim = out_dim

    def forward(self, x):
        z = self.encoder(x)                 # [B,64,1,1]
        return z.view(x.size(0), -1)       # [B,64]

# instantiate embedder + linear classifier
feature_dim = 64
embedder    = ConvEmbedder(feature_dim).to(device)
classifier  = nn.Linear(feature_dim, num_base).to(device)

sup_opt = optim.Adam(
    list(embedder.parameters()) + list(classifier.parameters()),
    lr=1e-3, weight_decay=1e-4
)

print(f"=== Supervised pre‑train ({SUPER_TUNE_EPOCHS} epochs) on {num_base} classes ===")
for epoch in range(SUPER_TUNE_EPOCHS):
    total_loss, total_acc, count = 0., 0., 0
    for xb, yb in tqdm(base_loader, desc=f"Pretrain E{epoch+1}", unit="batch"):
        xb, yb = xb.to(device), yb.to(device)
        zb     = embedder(xb)               # Eq.(1)
        logits = classifier(zb)
        loss   = F.cross_entropy(logits, yb)

        sup_opt.zero_grad(); loss.backward(); sup_opt.step()

        with torch.no_grad():
            preds      = logits.argmax(1)
            total_acc += (preds==yb).sum().item()
            total_loss+= loss.item() * xb.size(0)
            count    += xb.size(0)

    print(f"Pre‑Epoch {epoch+1}: Loss {total_loss/count:.4f}  Acc {total_acc/count:.4f}")

# 3.2 Freeze embedder unless META_FINE_TUNE
for p in embedder.parameters():
    p.requires_grad = META_FINE_TUNE

############################################
# 4. Meta‑Model Components
############################################
class TaskAggregator(nn.Module):
    def __init__(self, D, N, H=128):
        super().__init__()
        self.n_way, self.feat_dim = N, D
        self.label_emb = nn.Embedding(N,32)
        self.mlp       = nn.Sequential(
            nn.Linear(D+32,H), nn.ReLU(),
            nn.Linear(H,H),
        )
    def forward(self, zs, ys):
        y_emb = self.label_emb(ys.view(-1))
        h     = self.mlp(torch.cat([zs,y_emb],1))
        return h.view(self.n_way,-1,h.size(-1)).mean(1)

class HyperNet(nn.Module):
    def __init__(self, H, D, C):
        super().__init__()
        self.H, self.D, self.C = H, D, C
        self.net    = nn.Sequential(
            nn.Linear(H*C,256), nn.ReLU(),
            nn.Linear(256,256), nn.ReLU()
        )
        self.mu_head = nn.Linear(256, D*C)
        self.s_head  = nn.Linear(256, D*C)

    def forward(self, r):
        h  = self.net(r.view(-1))
        mu = self.mu_head(h).view(self.D, self.C)
        s  = self.s_head(h).view(self.D, self.C).clamp(-5,1)
        return mu, s

def sample_weights(mu, s, M_s=10):
    D,C = mu.shape
    eps = torch.randn(M_s, D, C, device=mu.device)
    return mu.unsqueeze(0) + torch.exp(s).unsqueeze(0)*eps

def predictive_probs(ws, zq):
    logits = torch.einsum('mdc,bd->mbc', ws, zq)
    return F.softmax(logits, dim=-1).mean(0)

############################################
# 5. Meta‑Train HyperNet
############################################
aggregator = TaskAggregator(feature_dim, N_way, H=128).to(device)
hypernet   = HyperNet(128, feature_dim, N_way).to(device)

opt = optim.Adam(
    list(filter(lambda p: p.requires_grad, embedder.parameters())) +
    list(aggregator.parameters()) +
    list(hypernet.parameters()),
    lr=1e-3, weight_decay=1e-4
)

print("=== Meta‑training HyperNet on few‑shot tasks ===")
for ep in tqdm(range(5000), desc="Meta‑Epoch"):
    (sx,sy),(qx,qy) = sample_meta_task_from_dict(
        meta_train_dict, N_way, K_shot, Q_query
    )
    zs     = embedder(sx)
    r      = aggregator(zs, sy)
    mu, s  = hypernet(r)
    ws     = sample_weights(mu, s, M_s=10)
    zq     = embedder(qx)
    p_pred = predictive_probs(ws, zq)
    loss   = F.nll_loss(p_pred.log(), qy.squeeze())

    opt.zero_grad(); loss.backward(); opt.step()

############################################
# 6. Save and Test
############################################
torch.save({
    "embedder":   embedder.state_dict(),
    "classifier": classifier.state_dict(),
    "aggregator": aggregator.state_dict(),
    "hypernet":   hypernet.state_dict(),
}, "conv_pretrain_and_meta.pth")
print("Saved full checkpoint.")

# Test on unseen
test_accs = []
for _ in range(200):
    (sx,sy),(qx,qy) = sample_meta_task_from_dict(
        meta_test_dict, N_way, K_shot, Q_query
    )
    zs     = embedder(sx)
    r      = aggregator(zs, sy)
    mu, s  = hypernet(r)
    ws     = sample_weights(mu, s, M_s=10)
    zq     = embedder(qx)
    p_pred = predictive_probs(ws, zq)
    test_accs.append((p_pred.argmax(1)==qy.squeeze()).float().mean().item())

mean_acc = np.mean(test_accs)*100
std_acc  = np.std(test_accs)*100
print(f"Test 5‑way/5‑shot: {mean_acc:.2f}% ± {std_acc:.2f}%")
