# import os, random, time
# import numpy as np
# from collections import defaultdict
# from PIL import Image
# from tqdm import tqdm

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

# from torchvision import transforms
# from torch.utils.data import Dataset, DataLoader

# # --------------- Config ----------------
# seed = 42
# N_way = 5
# K_shot = 5
# Q_query = 16
# feature_dim = 64
# embedder_layers = 4
# hypernet_layers = 2
# hypernet_hidden = 128
# META_EPOCHS = 500
# PRETRAIN_EPOCHS = 400
# BATCH_SIZE = 16
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
# random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

# # --------------- Dataset Preparation ---------------
# dataset_dir = "CUB_200_2011"
# images_file = os.path.join(dataset_dir, "images.txt")
# labels_file = os.path.join(dataset_dir, "image_class_labels.txt")
# image_dir   = os.path.join(dataset_dir, "images")

# print("loading dataset")
# def load_cub_images():
#     cub_images, cub_labels = [], []
#     with open(images_file) as f:
#         for line in f:
#             _, fn = line.strip().split()
#             cub_images.append(fn)
#     with open(labels_file) as f:
#         for line in f:
#             _, lb = line.strip().split()
#             cub_labels.append(int(lb))
#     assert len(cub_images)==len(cub_labels)
#     return cub_images, cub_labels

# def build_class_dict(cub_images, cub_labels, transform):
#     class_dict = defaultdict(list)
#     for fn, lb in zip(cub_images, cub_labels):
#         img = Image.open(os.path.join(image_dir, fn)).convert("RGB")
#         class_dict[lb].append(transform(img))
#     return class_dict

# def split_classes(labels, ratio=0.8):
#     uc = list(set(labels)); random.shuffle(uc)
#     n  = int(len(uc)*ratio)
#     return uc[:n], uc[n:]

# # Transforms
# pretrain_transform = transforms.Compose([
#     transforms.RandomResizedCrop(84),
#     transforms.RandomHorizontalFlip(),
#     transforms.ColorJitter(0.4, 0.4, 0.4),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
# ])
# standard_transform = transforms.Compose([
#     transforms.Resize((84,84)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
# ])

# print("loading images")
# cub_images, cub_labels = load_cub_images()
# print("done")
# print("splitting classes")
# train_cls, test_cls = split_classes(cub_labels)
# print("done")
# print("build dictionaries")
# full_class_dict = build_class_dict(cub_images, cub_labels, standard_transform)
# pretrain_class_dict = build_class_dict(cub_images, cub_labels, pretrain_transform)
# print("done")
# print("build train dict")
# meta_train_dict = {c: full_class_dict[c] for c in train_cls}
# meta_test_dict  = {c: full_class_dict[c] for c in test_cls}
# pretrain_dict   = {c: pretrain_class_dict[c] for c in train_cls}
# # Map training classes to contiguous labels for cross-entropy!
# train_class_list = sorted(list(train_cls))
# class_to_idx = {cls: i for i, cls in enumerate(train_class_list)}
# print("done")

# # # --------------- Pretrain Stage ---------------
# # class SimpleCUBDataset(Dataset):
# #     def __init__(self, class_dict, class_to_idx):
# #         self.samples = []
# #         for c in class_dict:
# #             for img in class_dict[c]:
# #                 self.samples.append((img, class_to_idx[c]))
# #     def __len__(self): return len(self.samples)
# #     def __getitem__(self, idx):
# #         x, y = self.samples[idx]
# #         return x, y

# # pretrain_dataset = SimpleCUBDataset(pretrain_dict, class_to_idx)
# # pretrain_loader = DataLoader(pretrain_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# class ConvEmbedder(nn.Module):
#     def __init__(self, out_dim, n_layers):
#         super().__init__()
#         layers, in_ch = [], 3
#         for _ in range(n_layers):
#             layers += [
#                 nn.Conv2d(in_ch, out_dim, 3, padding=1),
#                 nn.BatchNorm2d(out_dim),
#                 nn.ReLU(inplace=True),
#                 nn.MaxPool2d(2),
#             ]
#             in_ch = out_dim
#         self.encoder = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1))
#     def forward(self, x):
#         return self.encoder(x).view(x.size(0), -1)

# embedder = ConvEmbedder(out_dim=feature_dim, n_layers=embedder_layers).to(device)
# classifier = nn.Linear(feature_dim, len(train_class_list)).to(device)
# optimizer = optim.Adam(list(embedder.parameters()) + list(classifier.parameters()), lr=1e-3)
# criterion = nn.CrossEntropyLoss()

# # print("Pretraining Conv-4 embedder...")
# # for epoch in tqdm(range(PRETRAIN_EPOCHS)):
# #     losses, accs = [], []
# #     for x, y in pretrain_loader:
# #         x, y = x.to(device), y.to(device).long()
# #         z = embedder(x)
# #         logits = classifier(z)
# #         loss = criterion(logits, y)
# #         optimizer.zero_grad(); loss.backward(); optimizer.step()
# #         losses.append(loss.item())
# #         accs.append((logits.argmax(1) == y).float().mean().item())
# #     if (epoch+1) % 50 == 0:
# #         print(f"Epoch {epoch+1}/{PRETRAIN_EPOCHS} | Loss: {np.mean(losses):.4f} | Acc: {np.mean(accs)*100:.2f}%")

# # torch.save({'n_layers': embedder_layers, 'feature_dim': feature_dim, 'embedder_state_dict': embedder.state_dict()}, "pretrained_conv4d.pth")
# # print("Saved pretrained model to pretrained_conv4d.pth")

# # --------------- Meta-learning (HyperGP) ---------------
# def sample_meta_task(class_dict):
#     chosen = random.sample(list(class_dict), N_way)
#     sx, sy, qx, qy = [], [], [], []
#     for i,c in enumerate(chosen):
#         idx = random.sample(range(len(class_dict[c])), K_shot+Q_query)
#         for j in idx[:K_shot]: sx.append(class_dict[c][j]); sy.append(i)
#         for j in idx[K_shot:]: qx.append(class_dict[c][j]); qy.append(i)
#     return (torch.stack(sx).to(device), torch.tensor(sy,device=device).long()), \
#            (torch.stack(qx).to(device), torch.tensor(qy,device=device).long())

# class HyperNet(nn.Module):
#     def __init__(self, NK, D, H, n_hidden):
#         super().__init__()
#         layers = [nn.Linear(NK*(D+N_way), H), nn.ReLU(inplace=True)]
#         for _ in range(n_hidden-1): layers += [nn.Linear(H, H), nn.ReLU(inplace=True)]
#         self.net = nn.Sequential(*layers)
#         self.log_ell = nn.Linear(H, D)
#         self.log_sf  = nn.Linear(H, 1)
#         self.log_sn  = nn.Linear(H, 1)
#         nn.init.constant_(self.log_sn.bias, -3.0)
#     def forward(self, feats, labels_onehot):
#         inp = torch.cat([feats, labels_onehot], dim=1).view(-1)
#         h = self.net(inp)
#         ell = torch.exp(self.log_ell(h))
#         sf = torch.exp(self.log_sf(h)).squeeze()
#         sn = torch.exp(self.log_sn(h)).squeeze()
#         return ell, sf, sn

# def rbf_kernel(X1, X2, ell, sf):
#     diff = (X1.unsqueeze(1) - X2.unsqueeze(0)) / ell
#     return sf**2 * torch.exp(-0.5 * (diff**2).sum(-1))

# ckpt = torch.load("pretrained_conv4d.pth", map_location=device)
# embedder = ConvEmbedder(ckpt['feature_dim'], ckpt['n_layers']).to(device)
# embedder.load_state_dict(ckpt['embedder_state_dict'])
# embedder.train()

# hypernet = HyperNet(N_way*K_shot, feature_dim, hypernet_hidden, hypernet_layers).to(device)
# optimizer = optim.Adam(list(embedder.parameters()) + list(hypernet.parameters()), lr=1e-3)

# print("Training meta-learner...")
# for ep in tqdm(range(META_EPOCHS)):
#     (sx, sy), (qx, qy) = sample_meta_task(meta_train_dict)
#     z_s, z_q = embedder(sx), embedder(qx)
#     Y_s = F.one_hot(sy, N_way).float()
#     ell, sf, sn = hypernet(z_s, Y_s)
#     K_ss = rbf_kernel(z_s, z_s, ell, sf) + sn**2 * torch.eye(N_way*K_shot, device=device)
#     L = torch.cholesky(K_ss, upper=False)
#     alpha = torch.cholesky_solve(Y_s, L, upper=False)
#     mu_q = rbf_kernel(z_s, z_q, ell, sf).t() @ alpha
#     loss = F.cross_entropy(mu_q, qy)
#     optimizer.zero_grad(); loss.backward(); optimizer.step()

# # --------------- Evaluation ---------------
# print("Evaluating...")
# accs = []
# for _ in range(200):
#     (sx, sy), (qx, qy) = sample_meta_task(meta_test_dict)
#     z_s, z_q = embedder(sx), embedder(qx)
#     Y_s = F.one_hot(sy, N_way).float()
#     ell, sf, sn = hypernet(z_s, Y_s)
#     K_ss = rbf_kernel(z_s, z_s, ell, sf) + sn**2 * torch.eye(N_way*K_shot, device=device)
#     L = torch.cholesky(K_ss, upper=False)
#     alpha = torch.cholesky_solve(Y_s, L, upper=False)
#     mu_q = rbf_kernel(z_s, z_q, ell, sf).t() @ alpha
#     accs.append((mu_q.argmax(1) == qy).float().mean().item())
# mean, std = np.mean(accs), np.std(accs)
# print(f"\nTest Accuracy: {mean*100:.2f}% ± {1.96*std/np.sqrt(200)*100:.2f}%")

import os, random
import numpy as np
from collections import defaultdict
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms

# -------------------- Config --------------------
seed             = 42
N_way            = 5
K_shot           = 5
Q_query          = 16
feature_dim      = 64
embedder_layers  = 4
hypernet_layers  = 2
hypernet_hidden  = 128
META_EPOCHS      = 600
device           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        layers = [nn.Linear(NK*(D+N_way), H), nn.ReLU(inplace=True)]
        for _ in range(n_hidden-1):
            layers += [nn.Linear(H, H), nn.ReLU(inplace=True)]
        self.net    = nn.Sequential(*layers)
        self.log_ell = nn.Linear( H, D)
        self.log_sf  = nn.Linear( H, 1)
        self.log_sn  = nn.Linear( H, 1)
        nn.init.constant_(self.log_sn.bias, -3.0)
    def forward(self, feats, labels_onehot):
        # feats: [NK, D], labels_onehot: [NK, N_way]
        inp = torch.cat([feats, labels_onehot], dim=1).view(1,-1)
        h   = self.net(inp)
        # clamp into paper’s ranges:
        ell = 0.1  + 10.0*torch.sigmoid(self.log_ell(h))  # [0.1,10.1]
        sf  = 0.5  +  2.0*torch.sigmoid(self.log_sf( h))  # [0.5, 2.5]
        sn  = 1e-3 + 0.1*torch.sigmoid(self.log_sn( h))   # [0.001,0.101]
        return ell.view(-1), sf.view(-1), sn.view(-1)

def rbf_kernel(X1, X2, ell, sf):
    """
    X1: [n1, D]
    X2: [n2, D]
    ell: [D]        ← one length‐scale per feature dim
    sf: scalar      ← signal variance
    """
    # reshape ell so it broadcasts over the two sample dims:
    ell = ell.view(1, 1, -1)    # [1,1,D]
    diff = (X1.unsqueeze(1) - X2.unsqueeze(0)) / ell  # → [n1,n2,D]
    return sf**2 * torch.exp(-0.5 * (diff**2).sum(-1))  # [n1,n2]


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

# --------------- Load Pretrained Embedder ---------------
ckpt = torch.load("pretrained_conv4d.pth", map_location=device)
embedder = ConvEmbedder(ckpt['feature_dim'], ckpt['n_layers']).to(device)
embedder.load_state_dict(ckpt['embedder_state_dict'])
# embedder.eval()
embedder.train()  
# for p in embedder.parameters():
#     p.requires_grad = False

# --------------- Meta-Train HyperNet ---------------
hypernet = HyperNet(N_way*K_shot, feature_dim, hypernet_hidden, hypernet_layers).to(device)
opt       = optim.Adam(list(embedder.parameters()) + list(hypernet.parameters()), lr=1e-3)

print("Training meta-learner…")
for ep in tqdm(range(1, META_EPOCHS+1)):
    (sx,sy),(qx,qy) = sample_meta_task(meta_train_dict)
    with torch.no_grad():
        z_s = embedder(sx)   # [NK, D]
        z_q = embedder(qx)   # [Q_query*N_way, D]
    Y_s = F.one_hot(sy, N_way).float()

    ell, sf, sn = hypernet(z_s, Y_s)                # all shape [NK]
    K_ss = rbf_kernel(z_s, z_s, ell, sf)
    K_ss = rbf_kernel(z_s, z_s, ell, sf) \
     + (sn**2) * torch.eye(N_way*K_shot, device=device)
              # add noise diag
    K_sq = rbf_kernel(z_s, z_q, ell, sf)

    # <-- differentiable solve!
    alpha = torch.linalg.solve(K_ss, Y_s)           # [NK, N_way]
    mu_q  = K_sq.t() @ alpha                        # [Q*N_way, N_way]

    loss = F.cross_entropy(mu_q, qy)
    opt.zero_grad()
    loss.backward()
    opt.step()

    if ep % 100 == 0:
        print(f"[Episode {ep}/{META_EPOCHS}] Loss: {loss.item():.4f}")

# --------------- Evaluate -------------------
print("Evaluating…")
accs = []
for _ in range(600):
    (sx,sy),(qx,qy) = sample_meta_task(meta_test_dict)
    with torch.no_grad():
        z_s = embedder(sx)
        z_q = embedder(qx)
    Y_s = F.one_hot(sy, N_way).float()

    ell, sf, sn = hypernet(z_s, Y_s)
    K_ss = rbf_kernel(z_s, z_s, ell, sf) + torch.diag(sn**2 + 1e-4)
    K_sq = rbf_kernel(z_s, z_q, ell, sf)

    # solve + predict
    alpha = torch.linalg.solve(K_ss, Y_s)
    mu_q  = K_sq.t() @ alpha
    accs.append((mu_q.argmax(1)==qy).float().mean().item())

mean, std = np.mean(accs), np.std(accs)
print(f"Test Accuracy: {mean*100:.2f}% ± {1.96*std/np.sqrt(len(accs))*100:.2f}%")

# import os, random, time
# import numpy as np
# from collections import defaultdict
# from PIL import Image
# from tqdm import tqdm

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

# from torchvision import transforms

# # ------------------- Config -------------------
# seed = 42
# N_way      = 5
# K_shot     = 5
# Q_query    = 16
# feature_dim   = 64
# embedder_layers = 4
# hypernet_layers  = 2
# hypernet_hidden  = 128
# META_EPOCHS = 600
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)

# # ------------------- Dataset Preparation -------------------
# dataset_dir = "CUB_200_2011"
# images_file = os.path.join(dataset_dir, "images.txt")
# labels_file = os.path.join(dataset_dir, "image_class_labels.txt")
# image_dir   = os.path.join(dataset_dir, "images")

# def load_cub_images():
#     imgs, lbs = [], []
#     with open(images_file) as f:
#         for line in f:
#             _, fn = line.strip().split()
#             imgs.append(fn)
#     with open(labels_file) as f:
#         for line in f:
#             _, lb = line.strip().split()
#             lbs.append(int(lb))
#     return imgs, lbs

# def build_class_dict(imgs, lbs, transform):
#     d = defaultdict(list)
#     for fn, lb in zip(imgs, lbs):
#         img = Image.open(os.path.join(image_dir, fn)).convert("RGB")
#         d[lb].append(transform(img))
#     return d

# def split_classes(labels, ratio=0.8):
#     uc = list(set(labels))
#     random.shuffle(uc)
#     n = int(len(uc)*ratio)
#     return uc[:n], uc[n:]

# standard_transform = transforms.Compose([
#     transforms.Resize((84,84)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
# ])

# # load & split
# cub_images, cub_labels = load_cub_images()
# train_cls, test_cls = split_classes(cub_labels)
# full_dict = build_class_dict(cub_images, cub_labels, standard_transform)
# meta_train = {c: full_dict[c] for c in train_cls}
# meta_test  = {c: full_dict[c] for c in test_cls}

# def sample_task(class_dict):
#     chosen = random.sample(list(class_dict), N_way)
#     sx, sy, qx, qy = [], [], [], []
#     for i,c in enumerate(chosen):
#         assert len(class_dict[c]) >= K_shot+Q_query
#         idx = random.sample(range(len(class_dict[c])), K_shot+Q_query)
#         for j in idx[:K_shot]:
#             sx.append(class_dict[c][j]); sy.append(i)
#         for j in idx[K_shot:]:
#             qx.append(class_dict[c][j]); qy.append(i)
#     return ((torch.stack(sx).to(device),
#              torch.tensor(sy,device=device)),
#             (torch.stack(qx).to(device),
#              torch.tensor(qy,device=device)))

# # ------------------- Models -------------------
# class ConvEmbedder(nn.Module):
#     def __init__(self, out_dim, n_layers):
#         super().__init__()
#         layers, in_ch = [], 3
#         for _ in range(n_layers):
#             layers += [
#                 nn.Conv2d(in_ch, out_dim, 3, padding=1),
#                 nn.BatchNorm2d(out_dim),
#                 nn.ReLU(inplace=True),
#                 nn.MaxPool2d(2),
#             ]
#             in_ch = out_dim
#         self.encoder = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1))

#     def forward(self,x):
#         return self.encoder(x).view(x.size(0),-1)

# class HyperNet(nn.Module):
#     def __init__(self, D, N_way, H, n_hidden):
#         super().__init__()
#         # MLP to process each class’s prototype + one-hot
#         layers = [nn.Linear(D+N_way, H), nn.ReLU(inplace=True)]
#         for _ in range(n_hidden-1):
#             layers += [nn.Linear(H,H), nn.ReLU(inplace=True)]
#         self.net = nn.Sequential(*layers)

#         self.log_ell = nn.Linear(H, D)
#         self.log_sf  = nn.Linear(H, 1)
#         self.log_sn  = nn.Linear(H, 1)
#         nn.init.constant_(self.log_sn.bias, -3.0)

#     def forward(self, feats, labels_onehot):
#         """
#         feats:         [N_way*K_shot, D]
#         labels_onehot: [N_way*K_shot, N_way]
#         returns:
#           ell: [N_way, D]
#           sf:  [N_way]
#           sn:  [N_way]
#         """
#         D = feats.size(1)

#         # reshape into per-class blocks
#         feats = feats.view(N_way, K_shot, D)           # [N_way,K_shot,D]
#         labs  = labels_onehot.view(N_way, K_shot, N_way)  # [N_way,K_shot,N_way]

#         # class‐prototype
#         z_proto = feats.mean(dim=1)    # [N_way, D]
#         y_proto = labs[:,0,:]          # [N_way, N_way]  (they're identical per class)

#         cls_inp = torch.cat([z_proto, y_proto], dim=1)  # [N_way, D+N_way]
#         h = self.net(cls_inp)                          # [N_way, H]

#         ell = 0.1 + 10.0 * torch.sigmoid(self.log_ell(h))  # [N_way, D]
#         sf  = 0.5 +  2.0 * torch.sigmoid(self.log_sf(h))   # [N_way, 1]
#         sn  = 1e-3 + 0.1 * torch.sigmoid(self.log_sn(h))   # [N_way, 1]

#         return ell, sf.view(-1), sn.view(-1)


# def rbf_kernel(X1, X2, ell, sf):
#     # X1: [N1,D], X2: [N2,D], ell: [D], sf: scalar or [ ]
#     diff = (X1.unsqueeze(1) - X2.unsqueeze(0)) / ell
#     return sf**2 * torch.exp(-0.5*(diff**2).sum(-1))


# # ------------------- Load frozen embedder -------------------
# ckpt = torch.load("pretrained_conv4d.pth", map_location=device)
# embedder = ConvEmbedder(ckpt['feature_dim'], ckpt['n_layers']).to(device)
# embedder.load_state_dict(ckpt['embedder_state_dict'])
# embedder.eval()
# for p in embedder.parameters(): 
#     p.requires_grad = False

# # ------------------- Meta-training -------------------
# hypernet = HyperNet(feature_dim, N_way, hypernet_hidden, hypernet_layers).to(device)
# opt = optim.Adam(hypernet.parameters(), lr=1e-3)

# for ep in tqdm(range(1, META_EPOCHS+1)):
#     opt.zero_grad()
#     (sx, sy), (qx, qy) = sample_task(meta_train)
#     with torch.no_grad():
#         z_s = embedder(sx)   # [N_way*K_shot, D]
#         z_q = embedder(qx)   # [N_way*Q_query, D]
#     Y_s = F.one_hot(sy, N_way).float().to(device)  # [N_S, N_way]

#     # get per-class hyperparameters
#     ell, sf, sn = hypernet(z_s, Y_s)  # ell:[N_way,D], sf:[N_way], sn:[N_way]

#     N_S = z_s.size(0)
#     mu_q_list = []
#     # build one GP per class
#     for c in range(N_way):
#         ell_c = ell[c]
#         sf_c  = sf[c]
#         sn_c  = sn[c]

#         K_ss_c = (rbf_kernel(z_s, z_s, ell_c, sf_c) 
#                   + (sn_c**2 + 1e-6) * torch.eye(N_S, device=device))
#         K_sq_c = rbf_kernel(z_s, z_q, ell_c, sf_c)      # [N_S, N_Q]

#         L_c = torch.linalg.cholesky(K_ss_c)            # [N_S, N_S]
#         y_c = Y_s[:,c].unsqueeze(1)                    # [N_S, 1]

#         # solve α = (K_ss + σ_n²I)^{-1} y_c
#         alpha_c = torch.cholesky_solve(y_c, L_c)       # [N_S, 1]
#         mu_c    = K_sq_c.T @ alpha_c                   # [N_Q, 1]
#         mu_q_list.append(mu_c)

#     # stack per-class predictive means
#     mu_q = torch.cat(mu_q_list, dim=1)  # [N_Q, N_way]

#     loss = F.cross_entropy(mu_q, qy)
#     loss.backward()
#     opt.step()

#     if ep % 50 == 0:
#         print(f"[Ep {ep}] loss={loss.item():.4f}  ell_mean={ell.mean().item():.3f}")

# # ------------------- Evaluation -------------------
# print("Evaluating...")
# accs = []
# with torch.no_grad():
#     for _ in range(200):
#         (sx, sy), (qx, qy) = sample_task(meta_test)
#         z_s, z_q = embedder(sx), embedder(qx)
#         Y_s = F.one_hot(sy, N_way).float().to(device)
#         ell, sf, sn = hypernet(z_s, Y_s)

#         # same per-class loop as above
#         mu_q_list = []
#         for c in range(N_way):
#             ell_c, sf_c, sn_c = ell[c], sf[c], sn[c]
#             K_ss_c = rbf_kernel(z_s, z_s, ell_c, sf_c) + (sn_c**2+1e-6)*torch.eye(N_way*K_shot,device=device)
#             K_sq_c = rbf_kernel(z_s, z_q, ell_c, sf_c)
#             L_c = torch.linalg.cholesky(K_ss_c)
#             y_c = Y_s[:,c].unsqueeze(1)
#             alpha_c = torch.cholesky_solve(y_c, L_c)
#             mu_c = (K_sq_c.T @ alpha_c)
#             mu_q_list.append(mu_c)
#         mu_q = torch.cat(mu_q_list, dim=1)

#         accs.append((mu_q.argmax(1) == qy).float().mean().item())

# m, s = np.mean(accs), np.std(accs)
# print(f"Test acc: {m*100:.2f}% ± {1.96*s/np.sqrt(len(accs))*100:.2f}%")


