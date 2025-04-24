# import os, random, math
# import numpy as np
# import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
# from collections import defaultdict
# from torchvision import transforms
# from PIL import Image
# from tqdm import tqdm

# # — Device & seeds
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.manual_seed(42); np.random.seed(42); random.seed(42)

# ############################################
# # 1. Load CUB‑200‑2011 and split classes
# ############################################
# dataset_dir = "CUB_200_2011"
# images_file = os.path.join(dataset_dir, "images.txt")
# labels_file = os.path.join(dataset_dir, "image_class_labels.txt")
# image_dir   = os.path.join(dataset_dir, "images")

# transform = transforms.Compose([
#     transforms.Resize((84,84)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
# ])

# cub_images, cub_labels = [], []
# with open(images_file) as f:
#     for line in f:
#         _, fn = line.strip().split()
#         cub_images.append(fn)
# with open(labels_file) as f:
#     for line in f:
#         _, lb = line.strip().split()
#         cub_labels.append(int(lb))

# assert len(cub_images)==len(cub_labels)
# all_class_dict = defaultdict(list)
# for fn, lb in zip(cub_images, cub_labels):
#     img = Image.open(os.path.join(image_dir, fn)).convert("RGB")
#     all_class_dict[lb].append(transform(img))

# def split_classes(labels, ratio=0.8):
#     uc = list(set(labels)); random.shuffle(uc)
#     n  = int(len(uc)*ratio)
#     return uc[:n], uc[n:]

# train_cls, test_cls = split_classes(cub_labels, 0.8)
# meta_train_dict = {c:all_class_dict[c] for c in train_cls}
# meta_test_dict  = {c:all_class_dict[c] for c in test_cls}

# ############################################
# # 2. Meta‑task sampler
# ############################################
# def sample_meta_task(class_dict, N_way=5, K_shot=5, Q_query=15):
#     chosen = random.sample(list(class_dict), N_way)
#     sx, sy, qx, qy = [], [], [], []
#     for i,c in enumerate(chosen):
#         imgs = class_dict[c]
#         idx  = random.sample(range(len(imgs)), K_shot+Q_query)
#         for j in idx[:K_shot]:
#             sx.append(imgs[j]); sy.append(i)
#         for j in idx[K_shot:]:
#             qx.append(imgs[j]); qy.append(i)
#     sx = torch.stack(sx).to(device)
#     qx = torch.stack(qx).to(device)
#     sy = torch.tensor(sy,device=device).long()
#     qy = torch.tensor(qy,device=device).long()
#     return (sx,sy), (qx,qy)

# ############################################
# # 3. Embedding net: Conv‑4
# ############################################
# class ConvEmbedder(nn.Module):
#     def __init__(self, out_dim=64):
#         super().__init__()
#         layers=[]
#         in_ch=3
#         for _ in range(4):
#             layers += [
#                 nn.Conv2d(in_ch,64,3,padding=1),
#                 nn.BatchNorm2d(64),
#                 nn.ReLU(inplace=True),
#                 nn.MaxPool2d(2)
#             ]
#             in_ch=64
#         self.encoder = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1))
#         self.out_dim = out_dim

#     def forward(self,x):
#         h = self.encoder(x)           # [B,64,1,1]
#         return h.view(h.size(0),-1)   # [B,64]

# ############################################
# # 4. HyperNet ⇒ GP kernel hyperparams
# ############################################
# class HyperNet(nn.Module):
#     def __init__(self, NK, D, H=256):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(NK*(D+N_way), H),
#             nn.ReLU(),
#             nn.Linear(H,H),
#             nn.ReLU()
#         )
#         self.log_ell = nn.Linear(H, D)
#         self.log_sf  = nn.Linear(H, 1)
#         self.log_sn  = nn.Linear(H, 1)
#         nn.init.constant_(self.log_sn.bias, -3.0)

#     def forward(self, feats, labels_onehot):
#         B, D = feats.shape
#         inp = torch.cat([feats, labels_onehot], dim=1).view(-1)
#         h   = self.net(inp)
#         ell = torch.exp(self.log_ell(h))
#         sf  = torch.exp(self.log_sf(h)).squeeze()
#         sn  = torch.exp(self.log_sn(h)).squeeze()
#         return ell, sf, sn

# ############################################
# # 5. RBF kernel
# ############################################
# def rbf_kernel(X1, X2, ell, sf):
#     diff = (X1.unsqueeze(1) - X2.unsqueeze(0)) / ell
#     return sf**2 * torch.exp(-0.5 * (diff**2).sum(-1))

# ############################################
# # 6. Meta‑train loop with Cholesky from torch.*
# ############################################
# N_way, K_shot, Q_query = 5,5,15
# feature_dim = 64
# NK = N_way*K_shot

# embedder = ConvEmbedder(feature_dim).to(device)
# hypernet  = HyperNet(NK, feature_dim).to(device)

# opt = optim.Adam(
#     list(embedder.parameters()) + list(hypernet.parameters()),
#     lr=1e-3, weight_decay=1e-4
# )

# epochs = 2000
# for ep in tqdm(range(epochs)):
#     (sx,sy),(qx,qy) = sample_meta_task(meta_train_dict, N_way,K_shot,Q_query)
#     z_s = embedder(sx)     # [NK,D]
#     z_q = embedder(qx)     # [NQ,D]
#     Y_s = F.one_hot(sy, N_way).float()  # [NK,N_way]

#     ell, sf, sn = hypernet(z_s, Y_s)

#     # build & decompose
#     K_ss = rbf_kernel(z_s, z_s, ell, sf)
#     K_ss = K_ss + sn**2 * torch.eye(NK,device=device)
#     L    = torch.cholesky(K_ss, upper=False)

#     # solve α = (K_ss+σ_n²I)^-1 Y_s
#     alpha = torch.cholesky_solve(Y_s, L, upper=False)  # <- fixed

#     K_sq = rbf_kernel(z_s, z_q, ell, sf)
#     mu_q = K_sq.t() @ alpha

#     loss = F.cross_entropy(mu_q, qy)
#     opt.zero_grad(); loss.backward(); opt.step()

#     if ep % 10 == 0:
#         acc = (mu_q.argmax(1)==qy).float().mean()
#         print(f"Ep{ep}  Loss {loss:.3f}  Acc {acc:.3f}")

# ############################################
# # 7. Test
# ############################################
# test_accs=[]
# for _ in range(200):
#     (sx,sy),(qx,qy) = sample_meta_task(meta_test_dict, N_way,K_shot,Q_query)
#     z_s = embedder(sx); z_q = embedder(qx)
#     Y_s = F.one_hot(sy, N_way).float()
#     ell, sf, sn = hypernet(z_s, Y_s)

#     K_ss = rbf_kernel(z_s, z_s, ell, sf) + sn**2*torch.eye(NK,device=device)
#     L    = torch.cholesky(K_ss, upper=False)
#     alpha= torch.cholesky_solve(Y_s, L, upper=False)
#     K_sq = rbf_kernel(z_s, z_q, ell, sf)
#     mu_q = K_sq.t() @ alpha

#     preds = mu_q.argmax(1)
#     test_accs.append((preds==qy).float().mean().item())

# print(f"Test 5‑way/5‑shot: {np.mean(test_accs)*100:.2f}% ± {np.std(test_accs)*100:.2f}%")

import os, random, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from collections import defaultdict
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

#— Device & seeds
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42); np.random.seed(42); random.seed(42)

############################################
# 1. Load CUB‑200‑2011 and split classes
############################################
dataset_dir = "CUB_200_2011"
images_file = os.path.join(dataset_dir, "images.txt")
labels_file = os.path.join(dataset_dir, "image_class_labels.txt")
image_dir   = os.path.join(dataset_dir, "images")

transform = transforms.Compose([
    transforms.Resize((84,84)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
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

assert len(cub_images)==len(cub_labels)
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
# 2. Meta‑task sampler
############################################
def sample_meta_task(class_dict, N_way=5, K_shot=5, Q_query=15):
    chosen = random.sample(list(class_dict), N_way)
    sx, sy, qx, qy = [], [], [], []
    for i,c in enumerate(chosen):
        imgs = class_dict[c]
        idx  = random.sample(range(len(imgs)), K_shot+Q_query)
        for j in idx[:K_shot]:
            sx.append(imgs[j]); sy.append(i)
        for j in idx[K_shot:]:
            qx.append(imgs[j]); qy.append(i)
    sx = torch.stack(sx).to(device)
    qx = torch.stack(qx).to(device)
    sy = torch.tensor(sy,device=device).long()
    qy = torch.tensor(qy,device=device).long()
    return (sx,sy), (qx,qy)

############################################
# 3. Embedding net: Conv‑4
############################################
class ConvEmbedder(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        layers=[]; in_ch=3
        for _ in range(4):
            layers += [
                nn.Conv2d(in_ch,64,3,padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            ]
            in_ch=64
        self.encoder = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1))
        self.out_dim = out_dim
    def forward(self,x):
        h = self.encoder(x)           # [B,64,1,1]
        return h.view(h.size(0),-1)   # [B,64]

############################################
# 4. HyperNet ⇒ GP kernel hyperparams (exact from repo)
############################################
class HyperNet(nn.Module):
    def __init__(self, NK, D, H=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NK*(D+N_way), H),
            nn.ReLU(inplace=True),
            nn.Linear(H,H),
            nn.ReLU(inplace=True),
        )
        self.log_ell = nn.Linear(H, D)
        self.log_sf  = nn.Linear(H, 1)
        self.log_sn  = nn.Linear(H, 1)
        # repo bias-init:
        nn.init.constant_(self.log_sn.bias, -3.0)

    def forward(self, feats, labels_onehot):
        # feats: [NK,D], labels_onehot: [NK,N_way]
        inp = torch.cat([feats, labels_onehot],dim=1).view(-1)
        h   = self.net(inp)
        ell = torch.exp(self.log_ell(h))      # lengthscales [D]
        sf  = torch.exp(self.log_sf(h)).squeeze()  # signal std
        sn  = torch.exp(self.log_sn(h)).squeeze()  # noise std
        return ell, sf, sn

############################################
# 5. RBF kernel (exact)
############################################
def rbf_kernel(X1, X2, ell, sf):
    # X1: [NK,D], X2: [*,D], ell:[D], sf:scalar
    diff = (X1.unsqueeze(1) - X2.unsqueeze(0)) / ell
    return sf**2 * torch.exp(-0.5 * (diff**2).sum(-1))

############################################
# 6. Meta‑train loop
############################################
N_way, K_shot, Q_query = 5,5,15
feature_dim = 64
NK = N_way * K_shot

embedder = ConvEmbedder(feature_dim).to(device)
hypernet  = HyperNet(NK, feature_dim).to(device)

opt = optim.Adam(
    list(embedder.parameters()) + list(hypernet.parameters()),
    lr=1e-3,
    weight_decay=1e-4
)

META_EPOCHS = 2000
for ep in tqdm(range(META_EPOCHS), desc="Meta‑Epoch"):
    (sx,sy),(qx,qy) = sample_meta_task(meta_train_dict, N_way,K_shot,Q_query)
    z_s = embedder(sx)                   # [NK,D]
    z_q = embedder(qx)                   # [NQ,D]
    Y_s = F.one_hot(sy, N_way).float()   # [NK,N_way]

    ell, sf, sn = hypernet(z_s, Y_s)      # [D], scalar, scalar

    # build & decompose
    K_ss = rbf_kernel(z_s, z_s, ell, sf)
    K_ss = K_ss + sn**2 * torch.eye(NK,device=device)
    L    = torch.cholesky(K_ss, upper=False)

    # solve α = (K_ss)^-1 Y_s
    alpha    = torch.cholesky_solve(Y_s, L, upper=False)  # [NK,N_way]

    # predictive mean at z_q
    K_sq = rbf_kernel(z_s, z_q, ell, sf)              # [NK,NQ]
    μ_q  = K_sq.t() @ alpha                               # [NQ,N_way]

    loss = F.cross_entropy(μ_q, qy)
    opt.zero_grad(); loss.backward(); opt.step()

    if ep % 10 == 0:
        acc = (μ_q.argmax(1)==qy).float().mean().item()
        print(f"Ep{ep:4d}  Loss {loss:.3f}  Acc {acc:.3f}")

############################################
# 7. Test
############################################
test_accs = []
for _ in range(200):
    (sx,sy),(qx,qy) = sample_meta_task(meta_test_dict, N_way,K_shot,Q_query)
    z_s = embedder(sx); z_q = embedder(qx)
    Y_s = F.one_hot(sy, N_way).float()

    ell, sf, sn = hypernet(z_s, Y_s)
    K_ss = rbf_kernel(z_s, z_s, ell, sf) + sn**2*torch.eye(NK,device=device)
    L    = torch.cholesky(K_ss, upper=False)
    α    = torch.cholesky_solve(Y_s, L, upper=False)
    μ_q  = rbf_kernel(z_s, z_q, ell, sf).t() @ α

    preds = μ_q.argmax(1)
    test_accs.append((preds==qy).float().mean().item())

print(f"\nTest 5‑way/5‑shot: {np.mean(test_accs)*100:.2f}% ± {np.std(test_accs)*100:.2f}%")
