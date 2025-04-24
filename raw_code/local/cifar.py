import os, random, math
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# Set device and seeds for reproducibility.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

############################################
# 1. Load the CUB-200-2011 Dataset and Build a Class Dictionary
############################################

# Set the path to the extracted CUB dataset directory.
dataset_dir = "CUB_200_2011"  # <-- update this path to your CUB folder

# Files that list images and labels.
images_file = os.path.join(dataset_dir, "images.txt")
labels_file = os.path.join(dataset_dir, "image_class_labels.txt")
image_dir = os.path.join(dataset_dir, "images")

# Define transformations (resize to 84x84, convert to tensor, and normalize).
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
        # Each line format: "index filename"
        _, img_filename = line.strip().split()
        cub_images.append(img_filename)

# Read corresponding labels.
cub_labels = []
with open(labels_file, "r") as f:
    for line in f:
        _, label = line.strip().split()
        cub_labels.append(int(label))

# Verify we have the same number of images and labels.
assert len(cub_images) == len(cub_labels), "Mismatch between images and labels."

# Build a dictionary mapping each class label to a list of transformed images.
all_class_dict = defaultdict(list)
for img_filename, label in zip(cub_images, cub_labels):
    img_path = os.path.join(image_dir, img_filename)
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        continue
    if transform is not None:
        img = transform(img)
    all_class_dict[label].append(img)

all_classes = sorted(list(all_class_dict.keys()))
print(f"Total classes in CUB: {len(all_classes)}")

# Split classes: for example, 80% for meta-training, 20% for meta-testing.
def split_classes(labels_list, train_ratio=0.8):
    unique_classes = list(set(labels_list))
    random.shuffle(unique_classes)
    num_train = int(len(unique_classes) * train_ratio)
    return unique_classes[:num_train], unique_classes[num_train:]

meta_train_classes, meta_test_classes = split_classes(cub_labels, train_ratio=0.8)
meta_train_dict = {cls: all_class_dict[cls] for cls in meta_train_classes if cls in all_class_dict}
meta_test_dict  = {cls: all_class_dict[cls] for cls in meta_test_classes if cls in all_class_dict}

print(f"Meta‑train classes: {meta_train_classes}")
print(f"Meta‑test classes: {meta_test_classes}")

############################################
# 2. Meta‑Task Sampling Utility Function
############################################

def sample_meta_task_from_dict(class_dict, N_way=5, K_shot=5, Q_query=15):
    """
    Samples a meta‑task from a class dictionary.
    For each of N_way classes, sample K_shot support images and Q_query query images.
    
    Returns:
      (support_imgs, support_labels), (query_imgs, query_labels)
    where support_imgs is a tensor of shape (N_way*K_shot, C, H, W) and support_labels are relabeled 0...N_way-1.
    """
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
            support_labels.append(i)  # relabel to 0 ... N_way-1
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

# 3.1 CNN Feature Extractor (the "backbone")
class CNNFeatureExtractor(nn.Module):
    def __init__(self, output_dim=64):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)  # 84->42
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)  # 42->21
        self.fc = nn.Linear(64 * 21 * 21, output_dim)
        
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

# 3.2 Hypernetwork for Kernel Hyperparameters.
# Here we use a simple MLP-based hypernetwork that takes a pooled feature and produces
# Gaussian parameters for kernel hyperparameters.
class Hypernetwork(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=64):
        super(Hypernetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 4)  # outputs: [mu_ell, log_sigma_ell, mu_sigma, log_sigma_sigma]
        
    def forward(self, pooled_feature):
        h = F.relu(self.fc1(pooled_feature))
        out = self.fc2(h)  # shape: (batch, 4); batch here is expected to be 1.
        mu_ell = out[:, 0:1]
        log_sigma_ell = out[:, 1:2]
        mu_sigma = out[:, 2:3]
        log_sigma_sigma = out[:, 3:4]
        return mu_ell, log_sigma_ell, mu_sigma, log_sigma_sigma

############################################
# 4. Deep Kernel and GP Log Marginal Likelihood Functions
############################################

def deep_kernel(feature1, feature2, ell, sigma_f):
    """
    Computes the RBF kernel between feature1 (N x d) and feature2 (M x d).
    ell: lengthscale, sigma_f: signal variance.
    """
    diff = feature1.unsqueeze(1) - feature2.unsqueeze(0)
    dist_sq = torch.sum(diff**2, dim=-1)
    K = sigma_f**2 * torch.exp(-dist_sq / (2 * ell**2))
    return K

def gp_log_marginal_likelihood(K, y, sigma_n, jitter):
    """
    Computes the Gaussian Process log marginal likelihood.
    
    K: kernel matrix (N x N)
    y: targets (N x 1)
    sigma_n: noise standard deviation.
    jitter: small constant for numerical stability.
    """
    N = y.shape[0]
    Ky = K + (sigma_n**2) * torch.eye(N, device=K.device)
    L = torch.linalg.cholesky(Ky + jitter * torch.eye(N, device=K.device))
    alpha = torch.cholesky_solve(y, L)
    data_fit = -0.5 * torch.matmul(y.t(), alpha)
    complexity = - torch.sum(torch.log(torch.diag(L)))
    constant = - (N/2) * torch.log(torch.tensor(2 * math.pi, device=K.device))
    return (data_fit + complexity + constant).squeeze()

############################################
# 5. GP Loss for Meta‑Training (One‑vs‑Rest GP Regression)
############################################

def compute_task_loss(features_support, support_labels, ell, sigma_f, N_way):
    """
    Computes the averaged negative log marginal likelihood (GP) loss over N_way classes.
    For each class, creates binary targets: +1 for that class, -1 for others.
    """
    total_loss = 0.0
    for c in range(N_way):
        target = torch.where(support_labels == c,
                             torch.tensor(1.0, device=device),
                             torch.tensor(-1.0, device=device)).float()
        K = deep_kernel(features_support, features_support, ell, sigma_f)
        log_marg_lik = gp_log_marginal_likelihood(K, target, sigma_n=0.1, jitter=1e-4)
        total_loss += -log_marg_lik
    return total_loss / N_way

############################################
# 6. Initialize Models and Optimizer
############################################

feature_extractor = CNNFeatureExtractor(output_dim=64).to(device)
hypernet = Hypernetwork(input_dim=64, hidden_dim=64).to(device)
optimizer = optim.Adam(list(feature_extractor.parameters()) + list(hypernet.parameters()), lr=1e-3)

############################################
# 7. Meta‑Training Loop
############################################

num_meta_iterations = 1000  # Adjust as needed.
N_way = 5   # 5-way tasks.
K_shot = 5  # 5 support images per class.
Q_query = 15  # 15 query images per class (though our training loss uses only support set here).

print("Starting meta-training on CUB (meta‑train classes)...")
for iteration in range(num_meta_iterations):
    optimizer.zero_grad()
    
    # Sample a meta-task from the meta_train_dict.
    (support_imgs, support_labels), _ = sample_meta_task_from_dict(meta_train_dict, N_way=N_way, K_shot=K_shot, Q_query=Q_query)
    
    # Extract support features.
    features_support = feature_extractor(support_imgs)  # (N_way*K_shot, 64)
    # Optionally normalize features.
    features_support = F.normalize(features_support, dim=1)
    
    # Pool features (e.g. mean over support set).
    pooled_feature = torch.mean(features_support, dim=0, keepdim=True)  # (1, 64)
    
    # Use the hypernetwork to produce Gaussian parameters for kernel hyperparameters.
    mu_ell, log_sigma_ell, mu_sigma, log_sigma_sigma = hypernet(pooled_feature)
    sigma_ell = F.softplus(log_sigma_ell) + 1e-6
    sigma_sigma = F.softplus(log_sigma_sigma) + 1e-6
    
    # Reparameterization: sample hyperparameters.
    epsilon_ell = torch.randn_like(mu_ell)
    epsilon_sigma = torch.randn_like(mu_sigma)
    ell = torch.exp(mu_ell + sigma_ell * epsilon_ell)
    sigma_f = torch.exp(mu_sigma + sigma_sigma * epsilon_sigma)
    sigma_f = torch.clamp(sigma_f, max=1e1)
    
    loss = compute_task_loss(features_support, support_labels, ell.item(), sigma_f.item(), N_way)
    reg_loss = 1e-3 * (sigma_f**2).mean()
    loss = loss + reg_loss
    loss.backward()
    optimizer.step()
    
    if iteration % 200 == 0:
        print(f"Iter {iteration}: Loss = {loss.item():.3f}, sampled ell = {ell.item():.3f}, sigma_f = {sigma_f.item():.3f}")

############################################
# 8. Meta‑Testing: Evaluate on Unseen Classes (meta‑test)
############################################

def gp_predict(features_support, y_support, features_query, ell, sigma_f):
    """
    Performs GP regression prediction.
    
    features_support: (n_support, d)
    y_support: (n_support, 1) with binary targets (+1 or -1)
    features_query: (n_query, d)
    ell, sigma_f: kernel hyperparameters (scalars)
    Returns:
      predictive mean (n_query, 1) and variance (n_query, 1)
    """
    K = deep_kernel(features_support, features_support, ell, sigma_f)
    n_support = features_support.shape[0]
    Ky = K + (0.1**2) * torch.eye(n_support, device=K.device)
    L = torch.linalg.cholesky(Ky + 1e-6 * torch.eye(n_support, device=K.device))
    alpha = torch.cholesky_solve(y_support, L)
    K_star = deep_kernel(features_query, features_support, ell, sigma_f)
    mean = torch.matmul(K_star, alpha)
    v = torch.linalg.solve_triangular(L, K_star.t(), upper=False)
    var = sigma_f**2 - torch.sum(v**2, dim=0, keepdim=True).t()
    return mean, var

all_acc = []
num_meta_test_tasks = 1000
for _ in range(num_meta_test_tasks):
    (support_imgs, support_labels), (query_imgs, query_labels) = sample_meta_task_from_dict(meta_test_dict, N_way=N_way, K_shot=K_shot, Q_query=Q_query)
    
    features_support = feature_extractor(support_imgs)
    features_query = feature_extractor(query_imgs)
    features_support = F.normalize(features_support, dim=1)
    features_query = F.normalize(features_query, dim=1)
    
    pooled_support = torch.mean(features_support, dim=0, keepdim=True)
    mu_ell, log_sigma_ell, mu_sigma, log_sigma_sigma = hypernet(pooled_support)
    ell_pred = torch.exp(mu_ell)
    sigma_f_pred = torch.exp(mu_sigma)
    
    n_query = features_query.shape[0]
    pred_scores = torch.zeros((n_query, N_way), device=device)
    for c in range(N_way):
        target_support = torch.where(support_labels == c,
                                     torch.tensor(1.0, device=device),
                                     torch.tensor(-1.0, device=device)).float()
        mean_query, _ = gp_predict(features_support, target_support, features_query, ell_pred.item(), sigma_f_pred.item())
        pred_scores[:, c] = mean_query.squeeze()
    _, pred_classes = torch.max(pred_scores, dim=1)
    acc = (pred_classes.unsqueeze(1) == query_labels).float().mean() * 100
    all_acc.append(acc.item())

avg_acc = np.mean(all_acc)
print(f"\nMeta‑Test Average Accuracy on Unseen CUB Classes: {avg_acc:.2f}%")

############################################
# 9. Visualize Some Query Predictions from a Meta-Test Task
############################################

(support_imgs, support_labels), (query_imgs, query_labels) = sample_meta_task_from_dict(meta_test_dict, N_way=N_way, K_shot=K_shot, Q_query=Q_query)
features_support = feature_extractor(support_imgs)
features_query = feature_extractor(query_imgs)
features_support = F.normalize(features_support, dim=1)
features_query = F.normalize(features_query, dim=1)
pooled_support = torch.mean(features_support, dim=0, keepdim=True)
mu_ell, log_sigma_ell, mu_sigma, log_sigma_sigma = hypernet(pooled_support)
ell_pred = torch.exp(mu_ell)
sigma_f_pred = torch.exp(mu_sigma)

n_query = features_query.shape[0]
pred_scores = torch.zeros((n_query, N_way), device=device)
for c in range(N_way):
    target_support = torch.where(support_labels == c,
                                 torch.tensor(1.0, device=device),
                                 torch.tensor(-1.0, device=device)).float()
    mean_query, _ = gp_predict(features_support, target_support, features_query, ell_pred.item(), sigma_f_pred.item())
    pred_scores[:, c] = mean_query.squeeze()
_, pred_classes = torch.max(pred_scores, dim=1)
    
query_imgs_cpu = query_imgs.cpu()
plt.figure(figsize=(10, 4))
for i in range(min(10, n_query)):
    plt.subplot(2, 5, i+1)
    img = query_imgs_cpu[i].permute(1,2,0).numpy()
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.title(f"True: {query_labels[i].item()}\nPred: {pred_classes[i].item()}")
    plt.axis("off")
plt.suptitle("Meta-Test Query Predictions (CUB)")
plt.show()




