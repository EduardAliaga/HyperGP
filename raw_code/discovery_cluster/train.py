# #!/usr/bin/env python3

# import os, random, math, yaml, time
# import numpy as np
# from collections import defaultdict
# from PIL import Image
# from tqdm import tqdm

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

# import wandb

# from torchvision import transforms


# with open("config.yaml") as f:
#     cfg = yaml.safe_load(f)

# wandb.init(
#     project=cfg["project"],
#     config=cfg
# )
# config = wandb.config

# config.update({
#     "lr":            float(config.lr),
#     "weight_decay":  float(config.weight_decay),
#     "embedder_layers": int(config.embedder_layers),
#     "hypernet_layers":  int(config.hypernet_layers),
#     "feature_dim":     int(config.feature_dim),
#     "hypernet_hidden": int(config.hypernet_hidden),
#     "META_EPOCHS":     int(config.META_EPOCHS),
#     "meta_batch_size": int(config.meta_batch_size),
#     "Q_query":         int(config.Q_query),
#     "bn_momentum":   float(config.bn_momentum),
#     "dropout":       float(config.dropout),
#     "beta1":         float(config.beta1),
#     "beta2":         float(config.beta2),
# }, allow_val_change=True)


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# random.seed(config.seed)
# np.random.seed(config.seed)
# torch.manual_seed(config.seed)

# dataset_dir = "CUB_200_2011"
# images_file = os.path.join(dataset_dir, "images.txt")
# labels_file = os.path.join(dataset_dir, "image_class_labels.txt")
# image_dir   = os.path.join(dataset_dir, "images")

# transform = transforms.Compose([
#     transforms.Resize((84,84)),
#     transforms.RandomHorizontalFlip(),       # simple augmentation
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

# all_class_dict = defaultdict(list)
# for fn, lb in zip(cub_images, cub_labels):
#     img = Image.open(os.path.join(image_dir, fn)).convert("RGB")
#     all_class_dict[lb].append(transform(img))

# def split_classes(labels, ratio=0.8):
#     uc = list(set(labels)); random.shuffle(uc)
#     n  = int(len(uc)*ratio)
#     return uc[:n], uc[n:]

# train_cls, test_cls = split_classes(cub_labels, 0.8)
# meta_train_dict = {c: all_class_dict[c] for c in train_cls}
# meta_test_dict  = {c: all_class_dict[c] for c in test_cls}


# def sample_meta_task(class_dict, N_way, K_shot, Q_query):
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


# class ConvEmbedder(nn.Module):
#     def __init__(self, out_dim, n_layers, bn_mom, drop_p):
#         super().__init__()
#         layers, in_ch, ch = [], 3, out_dim    # <-- use out_dim here
#         for _ in range(n_layers):
#             layers += [
#                 nn.Conv2d(in_ch,   ch, 3, padding=1),
#                 nn.BatchNorm2d(ch, momentum=bn_mom),
#                 nn.ReLU(inplace=True),
#                 nn.MaxPool2d(2),
#                 nn.Dropout(drop_p),
#             ]
#             in_ch = ch
#         self.encoder = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1))
#         self.out_dim = out_dim

#     def forward(self, x):
#         h = self.encoder(x)
#         return h.view(h.size(0), -1)


# class HyperNet(nn.Module):
#     def __init__(self, NK, D, H, n_hidden):
#         super().__init__()
#         layers = [ nn.Linear(NK*(D+config.N_way), H), nn.ReLU(inplace=True) ]
#         for _ in range(n_hidden-1):
#             layers += [ nn.Linear(H, H), nn.ReLU(inplace=True) ]
#         self.net = nn.Sequential(*layers)

#         self.log_ell = nn.Linear(H, D)
#         self.log_sf  = nn.Linear(H, 1)
#         self.log_sn  = nn.Linear(H, 1)
#         nn.init.constant_(self.log_sn.bias, -3.0)

#     def forward(self, feats, labels_onehot):
#         inp = torch.cat([feats, labels_onehot], dim=1).view(-1)
#         h   = self.net(inp)
#         ell = torch.exp(self.log_ell(h))
#         sf  = torch.exp(self.log_sf(h)).squeeze()
#         sn  = torch.exp(self.log_sn(h)).squeeze() if config.learn_sigma_n else None
#         return ell, sf, sn

# def rbf_kernel(X1, X2, ell, sf):
#     diff = (X1.unsqueeze(1) - X2.unsqueeze(0)) / ell
#     return sf**2 * torch.exp(-0.5 * (diff**2).sum(-1))


# embedder = ConvEmbedder(
#     out_dim=config.feature_dim,
#     n_layers=config.embedder_layers,
#     bn_mom=config.bn_momentum,
#     drop_p=config.dropout
# ).to(device)

# hypernet = HyperNet(
#     NK=config.N_way*config.K_shot,
#     D=config.feature_dim,
#     H=config.hypernet_hidden,
#     n_hidden=config.hypernet_layers
# ).to(device)

# opt = optim.Adam(
#     list(embedder.parameters()) + list(hypernet.parameters()),
#     lr=config.lr,
#     weight_decay=config.weight_decay
# )

# wandb.watch(embedder, log="all", log_freq=100)
# wandb.watch(hypernet,  log="all", log_freq=100)


# NK = config.N_way * config.K_shot
# for ep in tqdm(range(config.META_EPOCHS), desc="Meta‑Epoch"):
#     meta_losses = []
#     meta_accs   = []
#     for _ in range(config.meta_batch_size):
#         (sx,sy),(qx,qy) = sample_meta_task(
#             meta_train_dict,
#             config.N_way, config.K_shot, config.Q_query
#         )
#         z_s = embedder(sx)
#         z_q = embedder(qx)
#         Y_s = F.one_hot(sy, config.N_way).float()

#         ell, sf, sn = hypernet(z_s, Y_s)
#         K_ss = rbf_kernel(z_s, z_s, ell, sf)
#         K_ss = K_ss + (sn**2 if sn is not None else 1e-6)*torch.eye(NK,device=device)
#         L    = torch.linalg.cholesky(K_ss)
#         alpha= torch.cholesky_solve(Y_s, L, upper=False)

#         mu_q = rbf_kernel(z_s, z_q, ell, sf).transpose(0,1) @ alpha
#         loss = F.cross_entropy(mu_q, qy)
#         acc  = (mu_q.argmax(1)==qy).float().mean().item()

#         meta_losses.append(loss)
#         meta_accs.append(acc)

#     # average across task‐batch
#     loss = torch.stack(meta_losses).mean()
#     acc  = np.mean(meta_accs)

#     opt.zero_grad()
#     loss.backward()
#     opt.step()

#     wandb.log({
#       "train/loss": loss.item(),
#       "train/acc":  acc,
#       "ell_mean":   ell.mean().item(),
#       "sn":         (sn.item() if sn is not None else 0.0),
#     }, step=ep)

# test_accs = []
# start = time.time()
# for _ in range(200):
#     (sx,sy),(qx,qy) = sample_meta_task(
#         meta_test_dict,
#         config.N_way, config.K_shot, config.Q_query
#     )
#     z_s = embedder(sx); z_q = embedder(qx)
#     Y_s = F.one_hot(sy, config.N_way).float()

#     ell, sf, sn = hypernet(z_s, Y_s)
#     K_ss = rbf_kernel(z_s,z_s,ell,sf) + (sn**2 if sn is not None else 1e-6)*torch.eye(NK,device=device)
#     L    = torch.linalg.cholesky(K_ss)
#     alpha= torch.cholesky_solve(Y_s, L, upper=False)
#     mu_q = rbf_kernel(z_s, z_q, ell, sf).transpose(0,1) @ alpha

#     test_accs.append((mu_q.argmax(1)==qy).float().mean().item())

# duration = time.time() - start
# mean, std = np.mean(test_accs), np.std(test_accs)

# wandb.log({
#   "test/acc_mean": mean,
#   "test/acc_std":  std,
#   "test/time_s":  duration
# }, step=config.META_EPOCHS)

# print(f"\nTest {config.N_way}‑way/{config.K_shot}‑shot: {100*mean:.2f}% ± {100*std:.2f}%")

import os, random, math, yaml, time
import numpy as np
from collections import defaultdict
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
    "lr":              float(config.lr),
    "weight_decay":    float(config.weight_decay),
    "embedder_layers": int(config.embedder_layers),
    "hypernet_layers": int(config.hypernet_layers),

    # Cast your new sweep params:
    "feature_dim":     int(config.feature_dim),
    "hypernet_hidden": int(config.hypernet_hidden),
    "N_way":           int(config.N_way),
    "K_shot":          int(config.K_shot),
    "Q_query":         int(config.Q_query),
}, allow_val_change=True)


# 3) Device & seeds
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(config.seed)
np.random.seed(config.seed)
# after wandb.init(...) and config.update(...)
n_embedder = config.embedder_layers
n_hypernet  = config.hypernet_layers

torch.manual_seed(config.seed)

# 4) Data loading (CUB‑200‑2011)
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

# 5) Meta‑task sampler
def sample_meta_task(class_dict, N_way, K_shot, Q_query):
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

# 6) Models
# class ConvEmbedder(nn.Module):
#     def __init__(self, out_dim, n_layers):
#         super().__init__()
#         layers, in_ch = [], 3
#         ch = out_dim                       # ← use out_dim instead of 64
#         for _ in range(n_layers):
#             layers += [
#                 nn.Conv2d(in_ch,   ch, 3, padding=1),
#                 nn.BatchNorm2d(ch),
#                 nn.ReLU(inplace=True),
#                 nn.MaxPool2d(2),
#             ]
#             in_ch = ch
#         self.encoder = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1))
#         self.out_dim = out_dim

#     def forward(self, x):
#         h = self.encoder(x)               # shape: (batch, out_dim, 1, 1)
#         return h.view(h.size(0), -1)      # shape: (batch, out_dim)

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
        layers = []
        # first block: input → H
        layers += [nn.Linear(NK*(D+config.N_way), H), nn.ReLU(inplace=True)]
        # additional hidden blocks
        for _ in range(n_hidden-1):  # if n_hidden=2, you get two Linear+ReLU
            layers += [nn.Linear(H, H), nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*layers)

        self.log_ell = nn.Linear(H, D)
        self.log_sf  = nn.Linear(H, 1)
        self.log_sn  = nn.Linear(H, 1)
        nn.init.constant_(self.log_sn.bias, -3.0)

    def forward(self, feats, labels_onehot):
        inp = torch.cat([feats, labels_onehot], dim=1).view(-1)
        h   = self.net(inp)
        ell = torch.exp(self.log_ell(h))
        sf  = torch.exp(self.log_sf(h)).squeeze()
        sn  = torch.exp(self.log_sn(h)).squeeze() if config.learn_sigma_n else None
        return ell, sf, sn

def rbf_kernel(X1, X2, ell, sf):
    diff = (X1.unsqueeze(1) - X2.unsqueeze(0)) / ell
    return sf**2 * torch.exp(-0.5 * (diff**2).sum(-1))

# Instantiate
pretrained_path = "conv_pretrain_and_meta.pth"  # your pretrained model file
# pretrained_ckpt = torch.load(pretrained_path, map_location=device)

# Extract pretrained model config
# pretrained_n_layers = pretrained_ckpt['n_layers']  # e.g., stored when saving
# pretrained_feature_dim = pretrained_ckpt['feature_dim']

# Update config accordingly (not tunable!)
# config.update({
#     "feature_dim": pretrained_feature_dim,
# }, allow_val_change=True)

# Instantiate embedder with pretrained architecture
embedder = ConvEmbedder(
    out_dim=config.feature_dim,
    n_layers=config.embedder_layers
).to(device)

# Load pretrained weights into embedder
# embedder.load_state_dict(pretrained_ckpt['embedder_state_dict'])

# Embedder parameters remain tunable:
embedder.train() 

hypernet = HyperNet(
    NK=config.N_way*config.K_shot,
    D=config.feature_dim,
    H=config.hypernet_hidden,
    n_hidden=config.hypernet_layers
).to(device)


opt = optim.Adam(
    list(embedder.parameters()) + list(hypernet.parameters()),
    lr=config.lr,
    weight_decay=config.weight_decay
)

wandb.watch(embedder, log="all", log_freq=100)
wandb.watch(hypernet,  log="all", log_freq=100)

# 7) Meta‑Train Loop
NK = config.N_way * config.K_shot
for ep in tqdm(range(config.META_EPOCHS), desc="Meta‑Epoch"):
    (sx,sy),(qx,qy) = sample_meta_task(meta_train_dict,
                                       config.N_way,
                                       config.K_shot,
                                       config.Q_query)
    z_s = embedder(sx)
    z_q = embedder(qx)
    Y_s = F.one_hot(sy, config.N_way).float()

    ell, sf, sn = hypernet(z_s, Y_s)
    K_ss = rbf_kernel(z_s, z_s, ell, sf)
    K_ss = K_ss + (sn**2 if config.learn_sigma_n else 1e-6)*torch.eye(NK,device=device)
    L    = torch.cholesky(K_ss, upper=False)
    alpha = torch.cholesky_solve(Y_s, L, upper=False)

    K_sq = rbf_kernel(z_s, z_q, ell, sf)
    mu_q = K_sq.t() @ alpha

    loss = F.cross_entropy(mu_q, qy)
    acc  = (mu_q.argmax(1)==qy).float().mean().item()

    opt.zero_grad(); loss.backward(); opt.step()

    # log to wandb
    wandb.log({
      "train/loss": loss.item(),
      "train/acc":  acc,
      "ell_mean":   ell.mean().item(),
      "sf_mean":    sf.mean().item(),   # <-- log sf!
      "sn":         sn.item() if sn is not None else 0.0,
    }, step=ep)

    # Periodic validation:
    if ep % 50 == 0:
        val_accs = []
        for _ in range(20):
            (sx,sy),(qx,qy) = sample_meta_task(meta_test_dict,
                                               config.N_way,
                                               config.K_shot,
                                               config.Q_query)
            z_s = embedder(sx); z_q = embedder(qx)
            Y_s = F.one_hot(sy, config.N_way).float()
            ell, sf, sn = hypernet(z_s, Y_s)
            K_ss = rbf_kernel(z_s, z_s, ell, sf)
            K_ss = K_ss + (sn**2 if config.learn_sigma_n else 1e-6)*torch.eye(NK,device=device)
            L    = torch.cholesky(K_ss, upper=False)
            alpha = torch.cholesky_solve(Y_s, L, upper=False)
            mu_q = rbf_kernel(z_s, z_q, ell, sf).t() @ alpha
            preds = mu_q.argmax(1)
            val_accs.append((preds==qy).float().mean().item())
        val_acc_mean = np.mean(val_accs)
        wandb.log({"val/acc": val_acc_mean}, step=ep)

# 8) Test
test_accs = []
start = time.time()
for _ in range(200):
    (sx,sy),(qx,qy) = sample_meta_task(meta_test_dict,
                                       config.N_way,
                                       config.K_shot,
                                       config.Q_query)
    z_s = embedder(sx); z_q = embedder(qx)
    Y_s = F.one_hot(sy, config.N_way).float()

    ell, sf, sn = hypernet(z_s, Y_s)
    K_ss = rbf_kernel(z_s,z_s,ell,sf) + (sn**2 if config.learn_sigma_n else 1e-6)*torch.eye(NK,device=device)
    L    = torch.cholesky(K_ss, upper=False)
    alpha = torch.cholesky_solve(Y_s, L, upper=False)
    mu_q  = rbf_kernel(z_s, z_q, ell, sf).t() @ alpha

    preds = mu_q.argmax(1)
    test_accs.append((preds==qy).float().mean().item())

duration = time.time() - start
mean, std = np.mean(test_accs), np.std(test_accs)

wandb.log({
    "train/loss": loss.item(),
    "train/acc":  acc,
    "ell_mean":   ell.mean().item(),
    "sf_mean":    sf.mean().item(),   # <-- log mean sf per episode
    "sn":         sn.item() if sn is not None else 0.0,
}, step=ep)

print(f"\nTest {config.N_way}‑way/{config.K_shot}‑shot: {mean*100:.2f}% ± {std*100:.2f}%")
