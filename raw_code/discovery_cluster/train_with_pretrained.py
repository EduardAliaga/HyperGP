import os, random, time
import numpy as np
from collections import defaultdict
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# --------------- Config ----------------
seed = 42
N_way = 5
K_shot = 5
Q_query = 16
feature_dim = 64
embedder_layers = 4
hypernet_layers = 2
hypernet_hidden = 128
META_EPOCHS = 500
PRETRAIN_EPOCHS = 400
BATCH_SIZE = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

# --------------- Dataset Preparation ---------------
dataset_dir = "CUB_200_2011"
images_file = os.path.join(dataset_dir, "images.txt")
labels_file = os.path.join(dataset_dir, "image_class_labels.txt")
image_dir   = os.path.join(dataset_dir, "images")

print("loading dataset")
def load_cub_images():
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
    return cub_images, cub_labels

def build_class_dict(cub_images, cub_labels, transform):
    class_dict = defaultdict(list)
    for fn, lb in zip(cub_images, cub_labels):
        img = Image.open(os.path.join(image_dir, fn)).convert("RGB")
        class_dict[lb].append(transform(img))
    return class_dict

def split_classes(labels, ratio=0.8):
    uc = list(set(labels)); random.shuffle(uc)
    n  = int(len(uc)*ratio)
    return uc[:n], uc[n:]

# Transforms
pretrain_transform = transforms.Compose([
    transforms.RandomResizedCrop(84),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
standard_transform = transforms.Compose([
    transforms.Resize((84,84)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

print("loading images")
cub_images, cub_labels = load_cub_images()
print("done")
print("splitting classes")
train_cls, test_cls = split_classes(cub_labels)
print("done")
print("build dictionaries")
full_class_dict = build_class_dict(cub_images, cub_labels, standard_transform)
pretrain_class_dict = build_class_dict(cub_images, cub_labels, pretrain_transform)
print("done")
print("build train dict")
meta_train_dict = {c: full_class_dict[c] for c in train_cls}
meta_test_dict  = {c: full_class_dict[c] for c in test_cls}
pretrain_dict   = {c: pretrain_class_dict[c] for c in train_cls}
# Map training classes to contiguous labels for cross-entropy!
train_class_list = sorted(list(train_cls))
class_to_idx = {cls: i for i, cls in enumerate(train_class_list)}
print("done")

# --------------- Pretrain Stage ---------------
class SimpleCUBDataset(Dataset):
    def __init__(self, class_dict, class_to_idx):
        self.samples = []
        for c in class_dict:
            for img in class_dict[c]:
                self.samples.append((img, class_to_idx[c]))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return x, y

pretrain_dataset = SimpleCUBDataset(pretrain_dict, class_to_idx)
pretrain_loader = DataLoader(pretrain_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

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

embedder = ConvEmbedder(out_dim=feature_dim, n_layers=embedder_layers).to(device)
classifier = nn.Linear(feature_dim, len(train_class_list)).to(device)
optimizer = optim.Adam(list(embedder.parameters()) + list(classifier.parameters()), lr=1e-3)
criterion = nn.CrossEntropyLoss()

print("Pretraining Conv-4 embedder...")
for epoch in tqdm(range(PRETRAIN_EPOCHS)):
    losses, accs = [], []
    for x, y in pretrain_loader:
        x, y = x.to(device), y.to(device).long()
        z = embedder(x)
        logits = classifier(z)
        loss = criterion(logits, y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        losses.append(loss.item())
        accs.append((logits.argmax(1) == y).float().mean().item())
    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1}/{PRETRAIN_EPOCHS} | Loss: {np.mean(losses):.4f} | Acc: {np.mean(accs)*100:.2f}%")

torch.save({'n_layers': embedder_layers, 'feature_dim': feature_dim, 'embedder_state_dict': embedder.state_dict()}, "pretrained_conv4d.pth")
print("Saved pretrained model to pretrained_conv4d.pth")

# --------------- Meta-learning (HyperGP) ---------------
def sample_meta_task(class_dict):
    chosen = random.sample(list(class_dict), N_way)
    sx, sy, qx, qy = [], [], [], []
    for i,c in enumerate(chosen):
        idx = random.sample(range(len(class_dict[c])), K_shot+Q_query)
        for j in idx[:K_shot]: sx.append(class_dict[c][j]); sy.append(i)
        for j in idx[K_shot:]: qx.append(class_dict[c][j]); qy.append(i)
    return (torch.stack(sx).to(device), torch.tensor(sy,device=device).long()), \
           (torch.stack(qx).to(device), torch.tensor(qy,device=device).long())

class HyperNet(nn.Module):
    def __init__(self, NK, D, H, n_hidden):
        super().__init__()
        layers = [nn.Linear(NK*(D+N_way), H), nn.ReLU(inplace=True)]
        for _ in range(n_hidden-1): layers += [nn.Linear(H, H), nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*layers)
        self.log_ell = nn.Linear(H, D)
        self.log_sf  = nn.Linear(H, 1)
        self.log_sn  = nn.Linear(H, 1)
        nn.init.constant_(self.log_sn.bias, -3.0)
    def forward(self, feats, labels_onehot):
        inp = torch.cat([feats, labels_onehot], dim=1).view(-1)
        h = self.net(inp)
        ell = torch.exp(self.log_ell(h))
        sf = torch.exp(self.log_sf(h)).squeeze()
        sn = torch.exp(self.log_sn(h)).squeeze()
        return ell, sf, sn

def rbf_kernel(X1, X2, ell, sf):
    diff = (X1.unsqueeze(1) - X2.unsqueeze(0)) / ell
    return sf**2 * torch.exp(-0.5 * (diff**2).sum(-1))

ckpt = torch.load("pretrained_conv4d.pth", map_location=device)
embedder = ConvEmbedder(ckpt['feature_dim'], ckpt['n_layers']).to(device)
embedder.load_state_dict(ckpt['embedder_state_dict'])
embedder.train()

hypernet = HyperNet(N_way*K_shot, feature_dim, hypernet_hidden, hypernet_layers).to(device)
optimizer = optim.Adam(list(embedder.parameters()) + list(hypernet.parameters()), lr=1e-3)

print("Training meta-learner...")
for ep in tqdm(range(META_EPOCHS)):
    (sx, sy), (qx, qy) = sample_meta_task(meta_train_dict)
    z_s, z_q = embedder(sx), embedder(qx)
    Y_s = F.one_hot(sy, N_way).float()
    ell, sf, sn = hypernet(z_s, Y_s)
    K_ss = rbf_kernel(z_s, z_s, ell, sf) + sn**2 * torch.eye(N_way*K_shot, device=device)
    L = torch.cholesky(K_ss, upper=False)
    alpha = torch.cholesky_solve(Y_s, L, upper=False)
    mu_q = rbf_kernel(z_s, z_q, ell, sf).t() @ alpha
    loss = F.cross_entropy(mu_q, qy)
    optimizer.zero_grad(); loss.backward(); optimizer.step()

# --------------- Evaluation ---------------
print("Evaluating...")
accs = []
for _ in range(200):
    (sx, sy), (qx, qy) = sample_meta_task(meta_test_dict)
    z_s, z_q = embedder(sx), embedder(qx)
    Y_s = F.one_hot(sy, N_way).float()
    ell, sf, sn = hypernet(z_s, Y_s)
    K_ss = rbf_kernel(z_s, z_s, ell, sf) + sn**2 * torch.eye(N_way*K_shot, device=device)
    L = torch.cholesky(K_ss, upper=False)
    alpha = torch.cholesky_solve(Y_s, L, upper=False)
    mu_q = rbf_kernel(z_s, z_q, ell, sf).t() @ alpha
    accs.append((mu_q.argmax(1) == qy).float().mean().item())
mean, std = np.mean(accs), np.std(accs)
print(f"\nTest Accuracy: {mean*100:.2f}% ± {1.96*std/np.sqrt(200)*100:.2f}%")
