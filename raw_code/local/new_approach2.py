import os
import random
import math
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from PIL import Image

# Set device and random seeds for reproducibility.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

############################################
# 1. Load CUB-200-2011 Dataset and Split Classes
############################################

dataset_dir = "CUB_200_2011"  # Update this to your path

images_file = os.path.join(dataset_dir, "images.txt")
labels_file = os.path.join(dataset_dir, "image_class_labels.txt")
image_dir = os.path.join(dataset_dir, "images")

transform = transforms.Compose([
    transforms.Resize((84,84)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

cub_images = []
with open(images_file, "r") as f:
    for line in f:
        _, fname = line.strip().split()
        cub_images.append(fname)

cub_labels = []
with open(labels_file, "r") as f:
    for line in f:
        _, label = line.strip().split()
        cub_labels.append(int(label))

assert len(cub_images) == len(cub_labels), "Mismatch between images and labels"

all_class_dict = defaultdict(list)
for fname, label in zip(cub_images, cub_labels):
    img_path = os.path.join(image_dir, fname)
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"Error loading {img_path}: {e}")
        continue
    if transform is not None:
        img = transform(img)
    all_class_dict[label].append(img)

all_classes = sorted(list(all_class_dict.keys()))
print(f"Total classes in CUB: {len(all_classes)}")

def split_classes(labels_list, train_ratio=0.8):
    unique = list(set(labels_list))
    random.shuffle(unique)
    num_train = int(len(unique) * train_ratio)
    return unique[:num_train], unique[num_train:]

meta_train_classes, meta_test_classes = split_classes(cub_labels, train_ratio=0.8)
meta_train_dict = {cls: all_class_dict[cls] for cls in meta_train_classes if cls in all_class_dict}
meta_test_dict  = {cls: all_class_dict[cls] for cls in meta_test_classes if cls in all_class_dict}

print(f"Meta-train classes: {meta_train_classes}")
print(f"Meta-test classes: {meta_test_classes}")

############################################
# 2. Meta-Task Sampling Function
############################################

def sample_meta_task_from_dict(class_dict, N_way=5, K_shot=5, Q_query=15):
    chosen_classes = random.sample(list(class_dict.keys()), N_way)
    support_imgs, support_labels = [], []
    query_imgs, query_labels = [], []
    
    for i, cls in enumerate(chosen_classes):
        imgs = class_dict[cls]
        if len(imgs) < K_shot + Q_query:
            indices = [random.randrange(len(imgs)) for _ in range(K_shot + Q_query)]
        else:
            indices = random.sample(range(len(imgs)), K_shot + Q_query)
        support_indices = indices[:K_shot]
        query_indices = indices[K_shot:]
        for idx in support_indices:
            support_imgs.append(imgs[idx])
            support_labels.append(i)  # relabel classes as 0...N_way-1
        for idx in query_indices:
            query_imgs.append(imgs[idx])
            query_labels.append(i)
    
    support_imgs = torch.stack(support_imgs).to(device)
    query_imgs = torch.stack(query_imgs).to(device)
    support_labels = torch.tensor(support_labels, dtype=torch.long, device=device).unsqueeze(1)
    query_labels = torch.tensor(query_labels, dtype=torch.long, device=device).unsqueeze(1)
    return (support_imgs, support_labels), (query_imgs, query_labels)

############################################
# 3. Define the Model Components
############################################

# 3.1 CNN Feature Extractor.
class CNNFeatureExtractor(nn.Module):
    def __init__(self, output_dim=64):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)  # 84 -> 42
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
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
        return x

# 3.2 Closed-form Adaptation via Ridge Regression.
def one_hot(labels, num_classes):
    n = labels.size(0)
    one_hot_labels = torch.zeros(n, num_classes, device=labels.device)
    one_hot_labels.scatter_(1, labels, 1)
    return one_hot_labels

def solve_ridge(X, Y, lam):
    d = X.size(1)
    A = X.t() @ X + lam * torch.eye(d, device=X.device)
    A_inv = torch.inverse(A)
    W = A_inv @ X.t() @ Y
    return W

############################################
# 4. Incorporate Kernel Hyperparameters with RFF
############################################

# Set mode: "analytically" uses deterministic (mean) hyperparameters,
# "bayesian" uses reparameterization (stochastic sampling).
mode = "bayesian"  # Change to "analytically" if desired.

def get_kernel_hyperparameters(pooled_feature, mode="analytically"):
    # A simple surrogate hypernetwork: two linear layers.
    fc1 = nn.Linear(64, 64).to(pooled_feature.device)
    fc2 = nn.Linear(64, 4).to(pooled_feature.device)
    h = F.relu(fc1(pooled_feature))
    out = fc2(h)  # (1, 4)
    mu_ell = out[:, 0:1]
    log_sigma_ell = out[:, 1:2]
    mu_sigma = out[:, 2:3]
    log_sigma_sigma = out[:, 3:4]
    
    if mode == "analytically":
        ell = torch.exp(mu_ell)
        sigma_f = torch.exp(mu_sigma)
    elif mode == "bayesian":
        epsilon_ell = torch.randn_like(mu_ell)
        epsilon_sigma = torch.randn_like(mu_sigma)
        sigma_ell = F.softplus(log_sigma_ell) + 1e-6
        sigma_sigma = F.softplus(log_sigma_sigma) + 1e-6
        ell = torch.exp(mu_ell + sigma_ell * epsilon_ell)
        sigma_f = torch.exp(mu_sigma + sigma_sigma * epsilon_sigma)
    else:
        raise ValueError("Mode must be either 'analytically' or 'bayesian'.")
    return ell, sigma_f

def rff_transform(X, ell, sigma_f, R):
    """
    Computes Random Fourier Features (RFF) to approximate an RBF kernel.
    X: (n, d) input features.
    ell: length-scale parameter.
    sigma_f: signal variance.
    R: number of random features.
    Returns: Z of shape (n, R)
    """
    n, d = X.shape
    # Sample random frequencies from N(0, I_d)
    omega = torch.randn(R, d, device=X.device) / ell  # Scale frequencies by 1/ell.
    # Sample random bias b uniformly in [0, 2*pi]
    b = 2 * math.pi * torch.rand(R, device=X.device)
    scale = math.sqrt(2 * (sigma_f**2) / R)
    Z = scale * torch.cos(X @ omega.t() + b)  # (n, R)
    return Z

############################################
# 5. Meta-Training Loop (Using RFF-based Kernel Ridge Regression)
############################################

N_way = 5   # 5-way classification.
K_shot = 5  # 5 support images per class.
Q_query = 15  # 15 query images per class.
lambda_reg = 1e-3  # Regularization parameter.
R_value = 50  # Number of Random Fourier Features.

# Initialize the feature extractor and optimizer.
feature_extractor = CNNFeatureExtractor(output_dim=64).to(device)
optimizer = optim.Adam(feature_extractor.parameters(), lr=1e-3)

num_meta_iterations = 2000
print("Starting meta-training on CUB with RFF-based classifier adaptation (mode: {})...".format(mode))
for iteration in range(num_meta_iterations):
    optimizer.zero_grad()
    (support_imgs, support_labels), (query_imgs, query_labels) = sample_meta_task_from_dict(
        meta_train_dict, N_way=N_way, K_shot=K_shot, Q_query=Q_query
    )
    # Compute support set features.
    X = feature_extractor(support_imgs)  # (N_way*K_shot, d)
    X = F.normalize(X, dim=1)
    
    # One-hot encode support labels.
    Y = one_hot(support_labels, N_way)
    
    # Compute a pooled feature from support set (mean)
    pooled_feature = torch.mean(X, dim=0, keepdim=True)  # (1, d)
    
    # Get kernel hyperparameters.
    ell, sigma_f = get_kernel_hyperparameters(pooled_feature, mode=mode)
    
    # Compute Random Fourier Features for the support set.
    Z = rff_transform(X, ell, sigma_f, R_value)  # (N_way*K_shot, R)
    
    # Solve for the task-specific classifier in the RFF space.
    W = solve_ridge(Z, Y, lambda_reg)  # W: (R, N_way)
    
    # Compute query features.
    Xq = feature_extractor(query_imgs)  # (N_way*Q_query, d)
    Xq = F.normalize(Xq, dim=1)
    # Compute RFF mapping for query set (using the same ell and sigma_f and same RFF parameters drawn anew)
    Zq = rff_transform(Xq, ell, sigma_f, R_value)  # (N_way*Q_query, R)
    
    # Compute logits in the RFF space.
    logits = Zq @ W  # (N_way*Q_query, N_way)
    query_labels_squeezed = query_labels.squeeze(1)
    loss = F.cross_entropy(logits, query_labels_squeezed)
    loss.backward()
    optimizer.step()
    
    if iteration % 200 == 0:
        print(f"Iteration {iteration}: Loss = {loss.item():.3f}")
        print(f"Sampled kernel hyperparameters: ell = {ell.item():.3f}, sigma_f = {sigma_f.item():.3f}")

############################################
# 6. Meta-Testing
############################################

def meta_test(feature_extractor, class_dict, N_way, K_shot, Q_query, lam, R_value, mode="bayesian"):
    all_acc = []
    num_tasks = 100
    for _ in range(num_tasks):
        (support_imgs, support_labels), (query_imgs, query_labels) = sample_meta_task_from_dict(
            class_dict, N_way=N_way, K_shot=K_shot, Q_query=Q_query
        )
        X = feature_extractor(support_imgs)
        X = F.normalize(X, dim=1)
        Y = one_hot(support_labels, N_way)
        pooled_feature = torch.mean(X, dim=0, keepdim=True)
        ell, sigma_f = get_kernel_hyperparameters(pooled_feature, mode=mode)
        Z = rff_transform(X, ell, sigma_f, R_value)
        W = solve_ridge(Z, Y, lam)
        
        Xq = feature_extractor(query_imgs)
        Xq = F.normalize(Xq, dim=1)
        Zq = rff_transform(Xq, ell, sigma_f, R_value)
        logits = Zq @ W
        preds = torch.argmax(logits, dim=1).unsqueeze(1)
        acc = (preds == query_labels).float().mean().item() * 100
        all_acc.append(acc)
    return np.mean(all_acc)

avg_test_acc = meta_test(feature_extractor, meta_test_dict, N_way, K_shot, Q_query, lambda_reg, R_value, mode=mode)
print(f"\nMeta-Test Average Accuracy on Unseen CUB Classes: {avg_test_acc:.2f}%")

############################################
# 7. Visualization of One Meta-Test Task
############################################

(support_imgs, support_labels), (query_imgs, query_labels) = sample_meta_task_from_dict(
    meta_test_dict, N_way=N_way, K_shot=K_shot, Q_query=Q_query
)
X = feature_extractor(support_imgs)
X = F.normalize(X, dim=1)
Y = one_hot(support_labels, N_way)
W = solve_ridge(rff_transform(X, *get_kernel_hyperparameters(torch.mean(X,dim=0,keepdim=True), mode=mode), R_value), Y, lambda_reg)

Xq = feature_extractor(query_imgs)
Xq = F.normalize(Xq, dim=1)
logits = rff_transform(Xq, *get_kernel_hyperparameters(torch.mean(X, dim=0, keepdim=True), mode=mode), R_value) @ W
pred_classes = torch.argmax(logits, dim=1)

query_imgs_cpu = query_imgs.cpu()
plt.figure(figsize=(12, 4))
n_query = query_imgs_cpu.size(0)
for i in range(min(10, n_query)):
    plt.subplot(2,5,i+1)
    img = query_imgs_cpu[i].permute(1,2,0).numpy()
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.title(f"True: {query_labels[i].item()}\nPred: {pred_classes[i].item()}")
    plt.axis("off")
plt.suptitle("Meta-Test Query Predictions on CUB")
plt.tight_layout()
plt.show()
