import os, random, math
import numpy as np
from collections import defaultdict
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, models
from torchvision.models import resnet18, ResNet18_Weights

#— Device & Seeds
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42); np.random.seed(42); random.seed(42)

#— Hyper‑params
META_FINE_TUNE  = False   # if True, allow layer4 of ResNet to be updated during meta‑train
N_way, K_shot, Q_query = 5, 5, 15
META_EPOCHS     = 2000
LR_EMBED        = 1e-5    # tiny if you fine‑tune embedder
LR_GP           = 1e-2

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
    cut = int(len(uc)*ratio)
    return uc[:cut], uc[cut:]

train_cls, test_cls = split_classes(cub_labels, 0.8)
meta_train_dict = {c: all_class_dict[c] for c in train_cls}
meta_test_dict  = {c: all_class_dict[c] for c in test_cls}

############################################
# 2. Meta‑Task Sampler
############################################
def sample_meta_task(class_dict):
    """Returns (sx,sy),(qx,qy) each: sx/qx [N*K or N*Q, C,H,W], sy/qy [N*K or N*Q,1]"""
    ways = random.sample(list(class_dict), N_way)
    sx,sy,qx,qy = [],[],[],[]
    for i,c in enumerate(ways):
        imgs = class_dict[c]
        idx  = random.sample(range(len(imgs)), K_shot+Q_query)
        for j in idx[:K_shot]:
            sx.append(imgs[j]); sy.append(i)
        for j in idx[K_shot:]:
            qx.append(imgs[j]); qy.append(i)
    sx = torch.stack(sx).to(device); sy = torch.tensor(sy,device=device).long().unsqueeze(1)
    qx = torch.stack(qx).to(device); qy = torch.tensor(qy,device=device).long().unsqueeze(1)
    return (sx,sy),(qx,qy)

############################################
# 3. Pretrained ResNet Embedder
############################################
class PretrainedResNetEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        b = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # keep conv1…layer4
        feat = list(b.children())[:-2]
        self.features = nn.Sequential(*feat, nn.AdaptiveAvgPool2d(1))
        self.out_dim  = b.fc.in_features  # 512

    def forward(self, x):
        z = self.features(x)         # [B,512,1,1]
        return z.view(x.size(0), -1) # [B,512]

embedder = PretrainedResNetEmbedder().to(device)
base_classes = sorted(meta_train_dict.keys())
num_base     = len(base_classes)
classifier = nn.Linear(embedder.out_dim, num_base).to(device)
ckpt = torch.load("resnet_pretrain_cub.pth", map_location=device)

# ❷ Restore the ResNet encoder
#    If you saved under key "embedder_state_dict":
embedder.load_state_dict(ckpt["embedder_state_dict"])

# ❸ (Optional) Restore the linear classifier head too
if "classifier_state_dict" in ckpt:
    classifier.load_state_dict(ckpt["classifier_state_dict"])

print("✅ Loaded bird‑pretrained ResNet + head from disk.")
# freeze all except layer4 if desired
for name,p in embedder.named_parameters():
    if META_FINE_TUNE:
        p.requires_grad = ("layer4" in name)
    else:
        p.requires_grad = False

for p in classifier.named_parameters():
    p.requires_grad = True

############################################
# 4. Gaussian‑Process Few‑Shot Head
############################################
class GPHead(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        # we'll learn log-lengthscale and log-noise
        self.log_ls    = nn.Parameter(torch.tensor(0.0))
        self.log_noise = nn.Parameter(torch.tensor(-3.0))
        self.feat_dim  = feat_dim

    def rbf_kernel(self, X1, X2):
        # X1: [n1,D], X2: [n2,D]
        diff = X1.unsqueeze(1) - X2.unsqueeze(0)  # [n1,n2,D]
        dist2= (diff*diff).sum(-1)                # [n1,n2]
        ls2  = torch.exp(2*self.log_ls)
        return torch.exp(-dist2/(2*ls2))

    def forward(self, Zs, Ys, Zq):
        """
        Zs: [N_way*K_shot, D]
        Ys: [N_way*K_shot,1] in {0..N_way-1}
        Zq: [N_way*Q_query, D]
        returns predictive class‑probabilities [Q, N_way]
        """
        N, K = N_way, K_shot
        Q     = Zq.size(0)
        # one‑hot support
        Nsamp = N*K
        Y1hot = F.one_hot(Ys.view(-1), num_classes=N).float()  # [NK,N]

        # compute K_ss + σ²I
        Kss = self.rbf_kernel(Zs, Zs)                          # [NK,NK]
        noise = torch.exp(2*self.log_noise)
        Kss = Kss + noise * torch.eye(Nsamp,device=device)

        # Cholesky solve α = Kss^{-1} Y1hot
        L    = torch.linalg.cholesky(Kss)                      # [NK,NK]
        α    = torch.cholesky_solve(Y1hot, L)                  # [NK,N]

        # compute K_qs [Q, NK], then preds = K_qs @ α [Q,N]
        Kqs  = self.rbf_kernel(Zq, Zs)                         # [Q,NK]
        f_q  = Kqs @ α                                         # [Q,N]
        return F.softmax(f_q, dim=-1)

gp_head = GPHead(embedder.out_dim).to(device)

############################################
# 5. Meta‑Train GPHead
############################################
opt = optim.Adam(
    list(filter(lambda p: p.requires_grad, embedder.parameters())) +
    list(gp_head.parameters()),
    lr=LR_EMBED
)
print("=== Meta‑training Gaussian‑Process few‑shot head ===")
for ep in tqdm(range(META_EPOCHS)):
    (sx,sy),(qx,qy) = sample_meta_task(meta_train_dict)
    zs = embedder(sx)     # [NK,D]
    zq = embedder(qx)     # [NQ,D]
    p_pred = gp_head(zs, sy, zq)       # [NQ,N_way]
    loss   = F.nll_loss(torch.log(p_pred), qy.view(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    if ep%10==0:
        acc = (p_pred.argmax(1)==qy.view(-1)).float().mean().item()
        print(f"Ep {ep:4d}  loss {loss:.3f}  acc {acc:.3f}")

############################################
# 6. Test on unseen classes
############################################
test_accs = []
for _ in range(200):
    (sx,sy),(qx,qy) = sample_meta_task(meta_test_dict)
    with torch.no_grad():
        p_pred = gp_head(embedder(sx), sy, embedder(qx))
        test_accs.append((p_pred.argmax(1)==qy.view(-1)).float().mean().item())

m,a = np.mean(test_accs)*100, np.std(test_accs)*100
print(f"\n*** Final Test 5‑way/5‑shot: {m:.2f}% ± {a:.2f}%")

