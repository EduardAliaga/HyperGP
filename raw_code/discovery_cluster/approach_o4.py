# import os, random
# import numpy as np
# import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
# from collections import defaultdict
# from torchvision import transforms
# from PIL import Image
from tqdm import tqdm


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.manual_seed(42); np.random.seed(42); random.seed(42)

# #— 1. Load CUB‑200‑2011
# dataset_dir = "CUB_200_2011"
# images_file = os.path.join(dataset_dir, "images.txt")
# labels_file = os.path.join(dataset_dir, "image_class_labels.txt")
# image_dir   = os.path.join(dataset_dir, "images")

# transform = transforms.Compose([
#     transforms.Resize((84,84)), transforms.ToTensor(),
#     transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
# ])

# cub_images, cub_labels = [], []
# with open(images_file) as f:
#     for line in f:
#         _, fn = line.strip().split(); cub_images.append(fn)
# with open(labels_file) as f:
#     for line in f:
#         _, lb = line.strip().split(); cub_labels.append(int(lb))

# assert len(cub_images)==len(cub_labels)
# all_class_dict = defaultdict(list)
# for fn, lb in zip(cub_images, cub_labels):
#     img = Image.open(os.path.join(image_dir, fn)).convert("RGB")
#     all_class_dict[lb].append(transform(img))

# # Split classes
# def split_classes(labels, ratio=0.8):
#     uc = list(set(labels)); random.shuffle(uc)
#     n = int(len(uc)*ratio); return uc[:n], uc[n:]
# train_cls, test_cls = split_classes(cub_labels, 0.8)
# meta_train_dict = {c:all_class_dict[c] for c in train_cls}
# meta_test_dict  = {c:all_class_dict[c] for c in test_cls}

# #— 2. Meta‑Task Sampler
# def sample_meta_task_from_dict(class_dict, N_way=5, K_shot=5, Q_query=15):
#     chosen = random.sample(list(class_dict), N_way)
#     sx, sy, qx, qy = [], [], [], []
#     for i,c in enumerate(chosen):
#         imgs = class_dict[c]
#         idx = random.sample(range(len(imgs)), K_shot+Q_query)
#         for j in idx[:K_shot]:
#             sx.append(imgs[j]); sy.append(i)
#         for j in idx[K_shot:]:
#             qx.append(imgs[j]); qy.append(i)
#     sx = torch.stack(sx).to(device)
#     qx = torch.stack(qx).to(device)
#     sy = torch.tensor(sy,device=device).long().unsqueeze(1)
#     qy = torch.tensor(qy,device=device).long().unsqueeze(1)
#     return (sx,sy), (qx,qy)

# #— 3. Model Components
# class ConvEmbedder(nn.Module):
#     def __init__(self, out_dim=64):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3,64,3,padding=1), nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2),
#             nn.Conv2d(64,64,3,padding=1), nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2),
#             nn.Conv2d(64,64,3,padding=1), nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2),
#             nn.Conv2d(64,64,3,padding=1), nn.ReLU(), nn.BatchNorm2d(64), nn.AdaptiveAvgPool2d(1),
#         )
#     def forward(self,x): return self.encoder(x).view(x.size(0),-1)

# class TaskAggregator(nn.Module):
#     def __init__(self, D, N, H=128):
#         super().__init__()
#         self.n_way, self.feat_dim = N, D
#         self.label_emb = nn.Embedding(N,32)
#         self.mlp = nn.Sequential(
#             nn.Linear(D+32,H), nn.ReLU(),
#             nn.Linear(H,H),
#         )
#     def forward(self, zs, ys):
#         y_emb = self.label_emb(ys.view(-1))
#         h     = self.mlp(torch.cat([zs,y_emb],1))
#         return h.view(self.n_way,-1,h.size(-1)).mean(1)

# class HyperNet(nn.Module):
#     def __init__(self, H, D, C):
#         super().__init__()
#         self.net = nn.Sequential(nn.Linear(H*C,256), nn.ReLU(),
#                                  nn.Linear(256,256), nn.ReLU())
#         self.mu_head = nn.Linear(256, D*C)
#         self.s_head  = nn.Linear(256, D*C)
#         self.H = H      # task‐embedding size per way
#         self.D = D      # feature dim
#         self.C = C 
#     def forward(self, r):
#         h  = self.net(r.view(-1))
#         mu = self.mu_head(h).view(self.D,self.C)
#         s  = self.s_head(h).view(self.D,self.C)
#         return mu, s

# def sample_weights(mu, s, M_s=10):
#     D,C = mu.shape
#     eps = torch.randn(M_s,D,C,device=mu.device)
#     return mu.unsqueeze(0) + torch.exp(s).unsqueeze(0)*eps

# def predictive_probs(ws, zq):
#     logits = torch.einsum('mdc,bd->mbc', ws, zq)  # [M_s,M_q,C]
#     return F.softmax(logits,dim=-1).mean(0)       # Eq.(8)

# #— 4. Instantiate & Train
# N_way, K_shot, Q_query = 5,5,15
# feature_dim, hidden_H = 64,128
# embedder   = ConvEmbedder(feature_dim).to(device)
# aggregator = TaskAggregator(feature_dim,N_way,hidden_H).to(device)
# hypernet   = HyperNet(hidden_H,feature_dim,N_way).to(device)
# opt = optim.Adam(list(embedder.parameters())
#                +list(aggregator.parameters())
#                +list(hypernet.parameters()), lr=1e-3)

# num_epochs = 5000
# for ep in tqdm(range(num_epochs)):
#     (sx,sy),(qx,qy) = sample_meta_task_from_dict(meta_train_dict,
#                                                 N_way,K_shot,Q_query)
#     zs = embedder(sx)                   # Eq.(1)
#     r  = aggregator(zs,sy)              # Eq.(2)
#     mu,s = hypernet(r)                  # Eq.(3–4)
#     ws    = sample_weights(mu,s,M_s=10) # Eq.(6)
#     zq    = embedder(qx)               # Eq.(7)
#     p_pred= predictive_probs(ws,zq)     # Eq.(8)
#     loss = F.nll_loss(p_pred.log(), qy.squeeze())  # Eq.(9)
#     opt.zero_grad(); loss.backward(); opt.step()    # Eq.(10)
#     if ep%500==0:
#         acc = (p_pred.argmax(1)==qy.squeeze()).float().mean()
#         print(f"Epoch {ep}  Loss {loss:.3f}  Acc {acc:.3f}")

# #— 5. Inference
# @torch.no_grad()
# def inference(sx,sy,qx):
#     zs = embedder(sx)
#     r  = aggregator(zs,sy)
#     mu,s = hypernet(r)
#     logits= embedder(qx) @ mu            # MAP estimate
#     return logits.softmax(-1)

import os, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from collections import defaultdict
from torchvision import transforms, models
from PIL import Image
from torchvision.models import ResNet18_Weights
#— Device & Seeds
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42); np.random.seed(42); random.seed(42)

#— 1. Load CUB‑200‑2011
dataset_dir = "CUB_200_2011"
images_file = os.path.join(dataset_dir, "images.txt")
labels_file = os.path.join(dataset_dir, "image_class_labels.txt")
image_dir   = os.path.join(dataset_dir, "images")

transform = transforms.Compose([
    transforms.Resize((84,84)), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

cub_images, cub_labels = [], []
with open(images_file) as f:
    for line in f:
        _, fn = line.strip().split(); cub_images.append(fn)
with open(labels_file) as f:
    for line in f:
        _, lb = line.strip().split(); cub_labels.append(int(lb))

assert len(cub_images)==len(cub_labels)
all_class_dict = defaultdict(list)
for fn, lb in zip(cub_images, cub_labels):
    img = Image.open(os.path.join(image_dir, fn)).convert("RGB")
    all_class_dict[lb].append(transform(img))

def split_classes(labels, ratio=0.8):
    uc = list(set(labels)); random.shuffle(uc)
    n = int(len(uc)*ratio); return uc[:n], uc[n:]
train_cls, test_cls = split_classes(cub_labels, 0.8)
meta_train_dict = {c:all_class_dict[c] for c in train_cls}
meta_test_dict  = {c:all_class_dict[c] for c in test_cls}

#— 2. Meta‑Task Sampler
def sample_meta_task_from_dict(class_dict, N_way=5, K_shot=5, Q_query=15):
    chosen = random.sample(list(class_dict), N_way)
    sx, sy, qx, qy = [], [], [], []
    for i,c in enumerate(chosen):
        imgs = class_dict[c]
        idx = random.sample(range(len(imgs)), K_shot+Q_query)
        for j in idx[:K_shot]:
            sx.append(imgs[j]); sy.append(i)
        for j in idx[K_shot:]:
            qx.append(imgs[j]); qy.append(i)
    sx = torch.stack(sx).to(device)
    qx = torch.stack(qx).to(device)
    sy = torch.tensor(sy,device=device).long().unsqueeze(1)
    qy = torch.tensor(qy,device=device).long().unsqueeze(1)
    return (sx,sy), (qx,qy)

#— 3. Model Components

# 3.1 Pretrained Embedder (ResNet‑18)

class PretrainedEmbedder(nn.Module):
    def __init__(self, fine_tune=False):
        super().__init__()
        # 1) Load full ResNet‑18 and chop off only the final FC:
        backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        # children() = [conv1, bn1, relu, maxpool,
        #              layer1, layer2, layer3, layer4,
        #              avgpool, fc]
        modules = list(backbone.children())[:-1]  # drop only fc
        self.encoder = nn.Sequential(*modules)    # now ends with avgpool
        self.out_dim = backbone.fc.in_features    # 512

        # 2) Freeze everything by default:
        for p in self.encoder.parameters():
            p.requires_grad = False

        # 3) If fine_tune, unfreeze just layer4 (the last residual block):
        if fine_tune:
            for name, param in backbone.named_parameters():
                # names like "layer4.0.conv1.weight", etc.
                if name.startswith("layer4"):
                    param.requires_grad = True
            # Note: we don't re-wrap encoder; layer4 is already in it

    def forward(self, x):
        # [B,3,84,84] → [B,512,1,1] → [B,512]
        z = self.encoder(x)
        return z.view(x.size(0), -1)

# 3.2 Task Aggregator (perm‑inv)
class TaskAggregator(nn.Module):
    def __init__(self, D, N, H=128):
        super().__init__()
        self.n_way, self.feat_dim = N, D
        self.label_emb = nn.Embedding(N,32)
        self.mlp = nn.Sequential(
            nn.Linear(D+32,H), nn.ReLU(),
            nn.Linear(H,H),
        )
    def forward(self, zs, ys):
        y_emb = self.label_emb(ys.view(-1))
        h     = self.mlp(torch.cat([zs,y_emb],1))
        # Eq. (2): mean over K-shot in each of N ways
        return h.view(self.n_way,-1,h.size(-1)).mean(1)

# 3.3 Hypernetwork → (μ,s)
class HyperNet(nn.Module):
    def __init__(self, H, D, C):
        super().__init__()
        self.H, self.D, self.C = H, D, C
        self.net = nn.Sequential(
            nn.Linear(H*C,256), nn.ReLU(),
            nn.Linear(256,256), nn.ReLU()
        )
        self.mu_head = nn.Linear(256, D*C)
        self.s_head  = nn.Linear(256, D*C)

    def forward(self, r):
        x  = r.view(-1)                      # [H*C]
        h  = self.net(x)                     # [256]
        mu = self.mu_head(h).view(self.D,self.C)
        s  = self.s_head(h).view(self.D,self.C)
        return mu, s                        # Eq. (3–4)

# 3.4 Sampling & Predictive
def sample_weights(mu, s, M_s=10):
    eps = torch.randn(M_s, mu.size(0), mu.size(1), device=mu.device)
    return mu.unsqueeze(0) + torch.exp(s).unsqueeze(0)*eps  # Eq. (6)

def predictive_probs(ws, zq):
    logits = torch.einsum('mdc,bd->mbc', ws, zq)  # [M_s,Q,C]
    return F.softmax(logits,dim=-1).mean(0)       # Eq. (8)

#— 4. Instantiate & Meta‑Train
N_way, K_shot, Q_query = 5, 5, 15
embedder   = PretrainedEmbedder(fine_tune=True).to(device)
feature_dim= embedder.out_dim  # 512
aggregator = TaskAggregator(feature_dim, N_way, 128).to(device)
hypernet   = HyperNet(128, feature_dim, N_way).to(device)

opt = optim.Adam(list(embedder.parameters())
               +list(aggregator.parameters())
               +list(hypernet.parameters()), lr=1e-3, weight_decay=1e-4)

num_epochs = 5000
for ep in tqdm(range(num_epochs)):
    (sx,sy),(qx,qy) = sample_meta_task_from_dict(meta_train_dict,
                                                N_way,K_shot,Q_query)
    # Eq. (1–2)
    zs = embedder(sx); r = aggregator(zs,sy)
    # Eq. (3–6)
    mu,s = hypernet(r); ws = sample_weights(mu,s,M_s=10)
    # Eq. (7–8)
    zq = embedder(qx); p_pred = predictive_probs(ws,zq)
    # Eq. (9–10)
    loss = F.nll_loss(p_pred.log(), qy.squeeze())
    opt.zero_grad(); loss.backward(); opt.step()
    if ep%500==0:
        acc = (p_pred.argmax(1)==qy.squeeze()).float().mean()
        print(f"[Epoch {ep:4d}] Loss {loss:.3f} Acc {acc:.3f}")

#— 5. Inference (MAP)
@torch.no_grad()
def inference(sx,sy,qx):
    zs = embedder(sx); r = aggregator(zs,sy)
    mu,_ = hypernet(r)
    logits = embedder(qx) @ mu               # single‑model
    return logits.softmax(-1)
