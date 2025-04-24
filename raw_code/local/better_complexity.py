import os, random, math
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms

# Set device and seeds.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

############################################
# 1. Load CUB-200-2011 Dataset and Split by Class
############################################

# Set the directory for the CUB dataset.
dataset_dir = "CUB_200_2011"  # <-- update this to your CUB directory

# Files: images.txt and image_class_labels.txt should be present in dataset_dir.
images_file = os.path.join(dataset_dir, "images.txt")
labels_file = os.path.join(dataset_dir, "image_class_labels.txt")
image_dir = os.path.join(dataset_dir, "images")

# Define image transformations.
transform = transforms.Compose([
    transforms.Resize((84,84)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Read image filenames.
cub_images = []
with open(images_file, "r") as f:
    for line in f:
        _, fname = line.strip().split()
        cub_images.append(fname)

# Read labels.
cub_labels = []
with open(labels_file, "r") as f:
    for line in f:
        _, label = line.strip().split()
        cub_labels.append(int(label))

assert len(cub_images) == len(cub_labels), "Mismatch in number of images and labels."

# Build a dictionary: class -> list of transformed images.
all_class_dict = defaultdict(list)
for fname, label in zip(cub_images, cub_labels):
    img_path = os.path.join(image_dir, fname)
    try:
        from PIL import Image
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"Error loading {img_path}: {e}")
        continue
    if transform is not None:
        img = transform(img)
    all_class_dict[label].append(img)

all_classes = sorted(list(all_class_dict.keys()))
print(f"Total classes in CUB: {len(all_classes)}")

# Split classes: 80% for meta-training, 20% for meta-testing.
def split_classes(labels_list, train_ratio=0.8):
    uniq = list(set(labels_list))
    random.shuffle(uniq)
    num_train = int(len(uniq)*train_ratio)
    return uniq[:num_train], uniq[num_train:]

meta_train_classes, meta_test_classes = split_classes(cub_labels, train_ratio=0.8)
meta_train_dict = {cls: all_class_dict[cls] for cls in meta_train_classes if cls in all_class_dict}
meta_test_dict  = {cls: all_class_dict[cls] for cls in meta_test_classes if cls in all_class_dict}

print(f"Meta-train classes: {meta_train_classes}")
print(f"Meta-test classes: {meta_test_classes}")

############################################
# 2. Meta-Task Sampling
############################################

def sample_meta_task_from_dict(class_dict, N_way=5, K_shot=5, Q_query=15):
    """
    Samples a meta-task with N_way classes.
    For each class, K_shot support images and Q_query query images are sampled.
    
    Returns:
      (support_imgs, support_labels), (query_imgs, query_labels)
    Tensors: support_imgs (N_way*K_shot, C, H, W); support_labels (N_way*K_shot, 1);
             query_imgs (N_way*Q_query, C, H, W); query_labels (N_way*Q_query, 1)
    """
    chosen_classes = random.sample(list(class_dict.keys()), N_way)
    support_imgs, support_labels = [], []
    query_imgs, query_labels = [], []
    for i, cls in enumerate(chosen_classes):
        imgs = class_dict[cls]
        if len(imgs) < K_shot + Q_query:
            indices = [random.randrange(len(imgs)) for _ in range(K_shot+Q_query)]
        else:
            indices = random.sample(range(len(imgs)), K_shot+Q_query)
        support_indices = indices[:K_shot]
        query_indices = indices[K_shot:]
        for idx in support_indices:
            support_imgs.append(imgs[idx])
            support_labels.append(i)  # relabel classes to 0...N_way-1.
        for idx in query_indices:
            query_imgs.append(imgs[idx])
            query_labels.append(i)
    support_imgs = torch.stack(support_imgs).to(device)
    query_imgs = torch.stack(query_imgs).to(device)
    support_labels = torch.tensor(support_labels, dtype=torch.long, device=device).unsqueeze(1)
    query_labels = torch.tensor(query_labels, dtype=torch.long, device=device).unsqueeze(1)
    return (support_imgs, support_labels), (query_imgs, query_labels)

############################################
# 3. Model Components
############################################

# 3.1 CNN Feature Extractor (backbone)
class CNNFeatureExtractor(nn.Module):
    def __init__(self, output_dim=64):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(3,32,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(32,32,kernel_size=3,padding=1)
        self.pool1 = nn.MaxPool2d(2)  # 84 -> 42
        self.conv3 = nn.Conv2d(32,64,kernel_size=3,padding=1)
        self.conv4 = nn.Conv2d(64,64,kernel_size=3,padding=1)
        self.pool2 = nn.MaxPool2d(2)  # 42 -> 21
        self.fc = nn.Linear(64*21*21, output_dim)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x  # shape: (batch, output_dim)

# 3.2 Hypernetwork H_ψ (produces Gaussian parameters for kernel hyperparameters)
class Hypernetwork(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=64):
        super(Hypernetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 4)  # outputs: [mu_ell, log_sigma_ell, mu_sigma, log_sigma_sigma]
    def forward(self, pooled_feature):
        h = F.relu(self.fc1(pooled_feature))
        out = self.fc2(h)
        mu_ell = out[:, 0:1]
        log_sigma_ell = out[:, 1:2]
        mu_sigma = out[:, 2:3]
        log_sigma_sigma = out[:, 3:4]
        return mu_ell, log_sigma_ell, mu_sigma, log_sigma_sigma

############################################
# 4. RFF and GP Functions
############################################

def rff_features(features, omega, b, sigma_f, R):
    """
    Computes Random Fourier Features:
      z(x) = sqrt(2 sigma_f^2 / R) * cos(features @ omega^T + b)
    features: (N, d), omega: (R, d), b: (R,)
    Returns: (N, R)
    """
    scale = math.sqrt(2 * (sigma_f**2) / R)
    z = scale * torch.cos(features @ omega.t() + b)
    return z

def compute_rff_regression_loss(z, y, lam=1e-3):
    """
    Given RFF features z (N x R) and targets y (N x 1), compute:
      A = z^T z + λI, then solve for w_mean = A^{-1}(z^T y)
    Loss = MSE + 0.5 logdet(A)
    """
    A = z.t() @ z + lam * torch.eye(z.shape[1], device=z.device)
    w_mean = torch.linalg.solve(A, z.t() @ y)
    y_pred = z @ w_mean
    mse_loss = torch.mean((y - y_pred)**2)
    sign, logdet = torch.linalg.slogdet(A)
    loss = mse_loss + 0.5 * logdet
    return loss, w_mean, torch.linalg.inv(A)

############################################
# 5. Meta-Training via RFF GP Loss
############################################

# Meta-task parameters.
N_way = 5     # 5-way classification.
K_shot = 5    # support images per class.
Q_query = 15  # query images per class.
R = 50        # number of RFF features.
lam = 1e-3    # regularization.

feature_extractor = CNNFeatureExtractor(output_dim=64).to(device)
hypernet = Hypernetwork(input_dim=64, hidden_dim=64).to(device)
optimizer = optim.Adam(list(feature_extractor.parameters()) + list(hypernet.parameters()), lr=1e-3)

num_meta_iterations = 2000
print("Starting meta-training on CUB using RFF GP loss...")

for iteration in range(num_meta_iterations):
    optimizer.zero_grad()
    (support_imgs, support_labels), _ = sample_meta_task_from_dict(meta_train_dict,
                                                                   N_way=N_way, K_shot=K_shot, Q_query=Q_query)
    features_support = feature_extractor(support_imgs)   # (N_way*K_shot, 64)
    pooled_feature = torch.mean(features_support, dim=0, keepdim=True)  # (1, 64)
    
    # Hypernetwork produces kernel hyperparameters.
    mu_ell, log_sigma_ell, mu_sigma, log_sigma_sigma = hypernet(pooled_feature)
    ell = torch.exp(mu_ell)
    sigma_f = torch.exp(mu_sigma)
    
    total_loss = 0.0
    for c in range(N_way):
        target = torch.where(support_labels == c,
                             torch.tensor(1.0, device=device),
                             torch.tensor(-1.0, device=device)).float()
        d = features_support.shape[1]  # 64
        omega = torch.randn(R, d, device=device) / ell  # (R, d)
        b = 2 * math.pi * torch.rand(R, device=device)    # (R,)
        z = rff_features(features_support, omega, b, sigma_f.item(), R)  # (N_way*K_shot, R)
        loss_c, _, _ = compute_rff_regression_loss(z, target, lam=lam)
        total_loss += loss_c
    loss_meta = total_loss / N_way
    loss_meta.backward()
    optimizer.step()
    
    if iteration % 200 == 0:
        print(f"Iter {iteration}: Meta Loss = {loss_meta.item():.3f}, ell = {ell.item():.3f}, sigma_f = {sigma_f.item():.3f}")

############################################
# 6. Meta-Testing via RFF Predictor on New Tasks
############################################

def rff_predict(features, omega, b, sigma_f, R, w_mean, A_inv):
    """
    Given features (N, d) and RFF parameters, predicts via:
      z(x) = sqrt(2 sigma_f^2 / R)*cos(features @ omega^T + b)
    Returns predictive mean and variance.
    """
    z_query = rff_features(features, omega, b, sigma_f, R)  # (N, R)
    mean = z_query @ w_mean
    var = torch.sum((z_query @ A_inv)**2, dim=1, keepdim=True)
    return mean, var

all_acc = []
num_meta_test_tasks = 100
for _ in range(num_meta_test_tasks):
    (support_imgs, support_labels), (query_imgs, query_labels) = sample_meta_task_from_dict(meta_test_dict,
                                                                                           N_way=N_way,
                                                                                           K_shot=K_shot,
                                                                                           Q_query=Q_query)
    features_support = feature_extractor(support_imgs)
    features_query = feature_extractor(query_imgs)
    pooled_support = torch.mean(features_support, dim=0, keepdim=True)
    mu_ell, log_sigma_ell, mu_sigma, log_sigma_sigma = hypernet(pooled_support)
    ell_pred = torch.exp(mu_ell)
    sigma_f_pred = torch.exp(mu_sigma)
    
    n_query = features_query.shape[0]
    pred_scores = torch.zeros((n_query, N_way), device=device)
    for c in range(N_way):
        target = torch.where(support_labels == c,
                             torch.tensor(1.0, device=device),
                             torch.tensor(-1.0, device=device)).float()
        d = features_support.shape[1]
        omega = torch.randn(R, d, device=device) / ell_pred
        b = 2 * math.pi * torch.rand(R, device=device)
        z_support = rff_features(features_support, omega, b, sigma_f_pred.item(), R)
        A = z_support.t() @ z_support + lam * torch.eye(R, device=device)
        w_mean = torch.linalg.solve(A, z_support.t() @ target)
        A_inv = torch.linalg.inv(A)
        mean_query, _ = rff_predict(features_query, omega, b, sigma_f_pred.item(), R, w_mean, A_inv)
        pred_scores[:, c] = mean_query.squeeze()
    _, pred_classes = torch.max(pred_scores, dim=1)
    acc = (pred_classes.unsqueeze(1) == query_labels).float().mean() * 100
    all_acc.append(acc.item())

avg_acc = np.mean(all_acc)
print(f"\nMeta-Test Average Accuracy on Unseen CUB Classes: {avg_acc:.2f}%")

############################################
# 7. Visualize Query Predictions for One Meta-Test Task
############################################

(support_imgs, support_labels), (query_imgs, query_labels) = sample_meta_task_from_dict(meta_test_dict,
                                                                                         N_way=N_way,
                                                                                         K_shot=K_shot,
                                                                                         Q_query=Q_query)
features_support = feature_extractor(support_imgs)
features_query = feature_extractor(query_imgs)
pooled_support = torch.mean(features_support, dim=0, keepdim=True)
mu_ell, log_sigma_ell, mu_sigma, log_sigma_sigma = hypernet(pooled_support)
ell_pred = torch.exp(mu_ell)
sigma_f_pred = torch.exp(mu_sigma)
n_query = features_query.shape[0]
pred_scores = torch.zeros((n_query, N_way), device=device)
for c in range(N_way):
    target = torch.where(support_labels == c,
                         torch.tensor(1.0, device=device),
                         torch.tensor(-1.0, device=device)).float()
    d = features_support.shape[1]
    omega = torch.randn(R, d, device=device) / ell_pred
    b = 2 * math.pi * torch.rand(R, device=device)
    z_support = rff_features(features_support, omega, b, sigma_f_pred.item(), R)
    A = z_support.t() @ z_support + lam * torch.eye(R, device=device)
    w_mean = torch.linalg.solve(A, z_support.t() @ target)
    A_inv = torch.linalg.inv(A)
    mean_query, _ = rff_predict(features_query, omega, b, sigma_f_pred.item(), R, w_mean, A_inv)
    pred_scores[:, c] = mean_query.squeeze()
_, pred_classes = torch.max(pred_scores, dim=1)
query_imgs_cpu = query_imgs.cpu()
plt.figure(figsize=(10,4))
for i in range(min(10, n_query)):
    plt.subplot(2,5,i+1)
    img = query_imgs_cpu[i].permute(1,2,0).numpy()
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.title(f"True: {query_labels[i].item()}\nPred: {pred_classes[i].item()}")
    plt.axis("off")
plt.suptitle("Meta-Test Query Predictions (CUB via RFF GP)")
plt.show()
