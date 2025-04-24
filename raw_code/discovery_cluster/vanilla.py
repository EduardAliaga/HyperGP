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

# Split classes
def split_classes(labels, ratio=0.8):
    uc = list(set(labels)); random.shuffle(uc)
    n = int(len(uc)*ratio); return uc[:n], uc[n:]
train_cls, test_cls = split_classes(cub_labels, 0.8)
meta_train_dict = {c:all_class_dict[c] for c in train_cls}
meta_test_dict  = {c:all_class_dict[c] for c in test_cls}

#— 2. Meta‑Task Sampler
def sample_meta_task_from_dict(class_dict, N_way=5, K_shot=1, Q_query=15):
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
class ConvEmbedder(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3,64,3,padding=1), nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2),
            nn.Conv2d(64,64,3,padding=1), nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2),
            nn.Conv2d(64,64,3,padding=1), nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2),
            nn.Conv2d(64,64,3,padding=1), nn.ReLU(), nn.BatchNorm2d(64), nn.AdaptiveAvgPool2d(1),
        )
    def forward(self,x): return self.encoder(x).view(x.size(0),-1)

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
        return h.view(self.n_way,-1,h.size(-1)).mean(1)

class HyperNet(nn.Module):
    def __init__(self, H, D, C):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(H*C,256), nn.ReLU(),
                                 nn.Linear(256,256), nn.ReLU())
        self.mu_head = nn.Linear(256, D*C)
        self.s_head  = nn.Linear(256, D*C)
        self.H = H      # task‐embedding size per way
        self.D = D      # feature dim
        self.C = C 
    def forward(self, r):
        h  = self.net(r.view(-1))
        mu = self.mu_head(h).view(self.D,self.C)
        s  = self.s_head(h).view(self.D,self.C)
        return mu, s

def sample_weights(mu, s, M_s=10):
    D,C = mu.shape
    eps = torch.randn(M_s,D,C,device=mu.device)
    return mu.unsqueeze(0) + torch.exp(s).unsqueeze(0)*eps

def predictive_probs(ws, zq):
    logits = torch.einsum('mdc,bd->mbc', ws, zq)  # [M_s,M_q,C]
    return F.softmax(logits,dim=-1).mean(0)       # Eq.(8)

#— 4. Instantiate & Train
N_way, K_shot, Q_query = 5,1,15
feature_dim, hidden_H = 64,128
embedder   = ConvEmbedder(feature_dim).to(device)
aggregator = TaskAggregator(feature_dim,N_way,hidden_H).to(device)
hypernet   = HyperNet(hidden_H,feature_dim,N_way).to(device)
opt = optim.Adam(list(embedder.parameters())
               +list(aggregator.parameters())
               +list(hypernet.parameters()), lr=1e-3)

num_epochs = 5000
for ep in tqdm(range(num_epochs)):
    (sx,sy),(qx,qy) = sample_meta_task_from_dict(meta_train_dict,
                                                N_way,K_shot,Q_query)
    zs = embedder(sx)                   # Eq.(1)
    r  = aggregator(zs,sy)              # Eq.(2)
    mu,s = hypernet(r)                  # Eq.(3–4)
    ws    = sample_weights(mu,s,M_s=10) # Eq.(6)
    zq    = embedder(qx)               # Eq.(7)
    p_pred= predictive_probs(ws,zq)     # Eq.(8)
    loss = F.nll_loss(p_pred.log(), qy.squeeze())  # Eq.(9)
    opt.zero_grad(); loss.backward(); opt.step()    # Eq.(10)
    if ep%500==0:
        acc = (p_pred.argmax(1)==qy.squeeze()).float().mean()
        print(f"Epoch {ep}  Loss {loss:.3f}  Acc {acc:.3f}")

#— 5. Inference
@torch.no_grad()
def inference(sx,sy,qx):
    zs = embedder(sx)
    r  = aggregator(zs,sy)
    mu,s = hypernet(r)
    logits= embedder(qx) @ mu            # MAP estimate
    return logits.softmax(-1)

# — after your training loop finishes —

# 6. Test on unseen classes
num_test_episodes = 200
test_accs = []

for _ in range(num_test_episodes):
    # Sample a fresh 5‑way/5‑shot task from meta_test_dict
    (sx, sy), (qx, qy) = sample_meta_task_from_dict(
        meta_test_dict, N_way, K_shot, Q_query
    )
    # 1) Embed support and aggregate
    zs = embedder(sx)                 # Eq.(1)
    r  = aggregator(zs, sy)           # Eq.(2)
    # 2) Generate posterior params + sample classifiers
    mu, s = hypernet(r)               # Eq.(3–4)
    ws    = sample_weights(mu, s, M_s=10)  # Eq.(6)
    # 3) Embed queries and compute predictive probs
    zq    = embedder(qx)              # Eq.(7)
    p_pred= predictive_probs(ws, zq)   # Eq.(8)
    # 4) Compute accuracy on this episode
    acc = (p_pred.argmax(1) == qy.squeeze()).float().mean()
    test_accs.append(acc.item())

# 5) Summarize
mean_acc = np.mean(test_accs) * 100
std_acc  = np.std(test_accs)  * 100
print(f"\nTest 5‑way/5‑shot accuracy over {num_test_episodes} episodes:")
print(f"  {mean_acc:.2f}%  ±  {std_acc:.2f}%")