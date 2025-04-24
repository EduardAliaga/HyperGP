import os, random
import math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from collections import defaultdict
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
from torchvision.models import resnet18, ResNet18_Weights

#— Device & Seeds
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42); np.random.seed(42); random.seed(42)

#— Hyper‑params
SUPER_TUNE_EPOCHS = 5    # epochs to train ConvEmbedder on base classes
META_FINE_TUNE     = True # True → allow ConvEmbedder last block update in meta‑train
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
class PretrainedResNetEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        # load ImageNet‑pretrained ResNet18
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # keep everything up through layer4, then global pool
        modules  = list(backbone.children())[:-2]   # conv1…layer3, layer4 blocks
        self.features = nn.Sequential(*modules,
                                      nn.AdaptiveAvgPool2d(1))
        self.out_dim = backbone.fc.in_features     # 512

    def forward(self, x):
        z = self.features(x)        # [B,512,1,1]
        return z.view(x.size(0), -1)  # [B,512]

# replace your ConvEmbedder + classifier
feature_dim = 512


embedder    = PretrainedResNetEmbedder().to(device)
classifier   = nn.Linear(feature_dim, num_base).to(device)

# optimizer trains both backbone & head
sup_opt = optim.Adam(
    list(embedder.parameters()) + list(classifier.parameters()),
    lr=1e-4,               # lower LR for finetuning
    weight_decay=1e-4
)

print(f"=== Supervised pre‑train ({SUPER_TUNE_EPOCHS} epochs) on {num_base} base classes ===")
for epoch in range(SUPER_TUNE_EPOCHS):
    total_loss, total_acc, count = 0., 0., 0
    for xb, yb in tqdm(base_loader, desc=f"Pretrain E{epoch+1}", unit="batch"):
        xb, yb = xb.to(device), yb.to(device)
        zb     = embedder(xb)                   # [B,512]
        logits = classifier(zb)                 # [B,num_base]
        loss   = F.cross_entropy(logits, yb)

        sup_opt.zero_grad()
        loss.backward()
        sup_opt.step()

        with torch.no_grad():
            preds      = logits.argmax(1)
            total_acc += (preds==yb).sum().item()
            total_loss+= loss.item()*xb.size(0)
            count    += xb.size(0)

    print(f"Pre‑Epoch {epoch+1}: Loss {total_loss/count:.4f}  Acc {total_acc/count:.4f}")
# — right after the supervised pre‑training loop finishes —
ckpt = {
    "embedder_state_dict":   embedder.state_dict(),
    "classifier_state_dict": classifier.state_dict(),
    "pretrain_epochs":       SUPER_TUNE_EPOCHS,
    "num_base_classes":      num_base,
}
torch.save(ckpt, "resnet_pretrain_cub.pth")
print("✅ Saved supervised‐trained ResNet checkpoint to resnet_pretrain_cub.pth")

# 3.2 Freeze everything except layer4 (and classifier) if META_FINE_TUNE=False
# after you build `embedder = PretrainedResNetEmbedder()`
# Freeze all of features *except* block #7 (which is layer4),
# when META_FINE_TUNE=True.  If META_FINE_TUNE=False, freeze all.