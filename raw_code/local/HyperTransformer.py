import os
import random
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

###############################
# 1. Utility functions for splitting the CUB classes
###############################
def split_classes(all_labels, train_ratio=0.8):
    """
    Split unique classes into train and test sets.
    
    Args:
        all_labels (list[int]): List of integer class labels.
        train_ratio (float): Ratio of classes to be used for training.
    
    Returns:
        (train_classes, test_classes): Tuple of lists.
    """
    unique_classes = list(set(all_labels))
    random.shuffle(unique_classes)
    num_train = int(len(unique_classes) * train_ratio)
    train_classes = unique_classes[:num_train]
    test_classes = unique_classes[num_train:]
    return train_classes, test_classes

###############################
# 2. Custom Dataset for CUB in a Meta-Learning Setting
###############################
class CubMetaDataset(Dataset):
    """
    A PyTorch Dataset for few-shot learning using the CUB dataset.
    It reads the file names and corresponding class labels from the provided text files,
    splits classes into train/test according to a specified ratio, and on each call samples
    an episodic task (support and query sets).
    
    Args:
        dataset_dir (str): Path to the extracted 'CUB_200_2011' folder.
        split (str): Either 'train' or 'test' to choose which classes to use.
        num_classes (int): Number of classes per episode (N-way).
        num_support (int): Number of support images per class (K-shot).
        num_query (int): Number of query images per class.
        transform: torchvision transforms to apply to images.
        train_ratio (float): Ratio of classes to use for training split.
    """
    def __init__(self, dataset_dir, split='train', num_classes=5, num_support=1, num_query=5,
                 transform=None, train_ratio=0.8):
        self.dataset_dir = dataset_dir
        self.images_file = os.path.join(dataset_dir, "images.txt")
        self.labels_file = os.path.join(dataset_dir, "image_class_labels.txt")
        self.image_dir = os.path.join(dataset_dir, "images")
        self.num_classes = num_classes
        self.num_support = num_support
        self.num_query = num_query
        self.transform = transform

        # Read image filenames.
        self.image_names = []
        with open(self.images_file, "r") as f:
            for line in f:
                # Each line: "index image_filename"
                _, img_filename = line.strip().split()
                self.image_names.append(img_filename)

        # Read image labels.
        self.labels = []
        with open(self.labels_file, "r") as f:
            for line in f:
                # Each line: "index label"
                _, label = line.strip().split()
                self.labels.append(int(label))
        
        # Ensure consistency.
        assert len(self.image_names) == len(self.labels), "Mismatch between images and labels."

        # Split the unique classes into train and test.
        self.train_classes, self.test_classes = split_classes(self.labels, train_ratio=train_ratio)
        if split == 'train':
            self.classes = self.train_classes
        else:
            self.classes = self.test_classes
        
        # Create a mapping from class -> list of image indices.
        self.class_to_indices = {cls: [] for cls in self.classes}
        for idx, label in enumerate(self.labels):
            if label in self.classes:
                self.class_to_indices[label].append(idx)
        
        # We set the length arbitrarily; each __getitem__ returns a newly sampled task.
        self.episode_length = 100000

    def __len__(self):
        return self.episode_length

    def __getitem__(self, index):
        # Randomly sample num_classes_per_task classes.
        chosen_classes = random.sample(self.classes, self.num_classes)
        support_set = []
        query_set = []
        for new_label, cls in enumerate(chosen_classes):
            indices = self.class_to_indices[cls]
            # Randomly sample num_support + num_query examples for this class.
            if len(indices) < self.num_support + self.num_query:
                # If not enough images, sample with replacement.
                chosen_indices = random.choices(indices, k=self.num_support + self.num_query)
            else:
                chosen_indices = random.sample(indices, self.num_support + self.num_query)
            support_indices = chosen_indices[:self.num_support]
            query_indices = chosen_indices[self.num_support:]
            
            for i in support_indices:
                img_path = os.path.join(self.image_dir, self.image_names[i])
                img = Image.open(img_path).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                # Append only once.
                support_set.append(img)
            
            for i in query_indices:
                img_path = os.path.join(self.image_dir, self.image_names[i])
                img = Image.open(img_path).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                query_set.append(img)
        # Generate labels: one label per support example.
        support_labels = [i for i in range(self.num_classes) for _ in range(self.num_support)]
        query_labels = [i for i in range(self.num_classes) for _ in range(self.num_query)]
        support_images = torch.stack(support_set, dim=0)
        query_images = torch.stack(query_set, dim=0)
        support_labels = torch.tensor(support_labels)
        query_labels = torch.tensor(query_labels)
        return support_images, support_labels, query_images, query_labels


###############################
# 3. Define Image Transformations
###############################
transform = transforms.Compose([
    transforms.Resize((84,84)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225]),
])

###############################
# 4. Define the Base Model (4-layer CNN) 
###############################
# class BaseModel(nn.Module):
#     """
#     A simple 4-layer CNN for few-shot classification.
#     """
#     def __init__(self, num_classes=5):
#         super(BaseModel, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
#         # self.conv4 = nn.Sequential(
#         #     nn.Conv2d(64, 64, kernel_size=3, padding=1),
#         #     nn.BatchNorm2d(64),
#         #     nn.ReLU(),
#         #     nn.MaxPool2d(2)
#         # )
#         # For images resized to 84x84, after four 2×2 poolings, spatial dims ~5×5.
#         self.feature_dim = 64 * 5 * 5
#         self.feature_dim = 64 * 10 * 10
#         self.classifier = nn.Linear(self.feature_dim, num_classes)
    
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.conv2(out)
#         out = self.conv3(out)
#         # out = self.conv4(out)
#         out = out.view(out.size(0), -1)
#         logits = self.classifier(out)
#         return logits

class BaseModel(nn.Module):
    """
    A simple 4-layer CNN for few-shot classification with a learnable projection layer.
    """
    def __init__(self, num_classes=5, projection_dim=130):
        super(BaseModel, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # After three 2x2 poolings on an 84x84 image, the spatial dims become roughly 10x10.
        self.feature_dim = 64 * 5 * 5
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        # Learnable projection layer: projects features to the desired input_dim for the hypernetwork.
        self.projection = nn.Linear(self.feature_dim, projection_dim)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        return logits

    def extract_features(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = out.view(out.size(0), -1)
        proj_features = self.projection(out)
        return proj_features

###############################
# 7. Updated Helper Function (removed on-the-fly projection)
###############################
# Removed get_support_features function.
# Instead, we call meta_model.extract_features(support_images) in the inner loop.

###############################
# 8. Updated Inner-Loop Adaptation (Support Set)
###############################
def inner_loop_adaptation(meta_model, hypernet, support_images, support_labels, T=3, inner_lr=1.0):
    """
    Adapt the meta_model to the current task using the support set.
    Updates only the classifier parameters using hypernetwork-generated updates.
    
    Args:
        meta_model: The universal model.
        hypernet: The hypernetwork.
        support_images: Tensor of support images.
        support_labels: Tensor of support labels.
        T: Number of inner-loop iterations.
        inner_lr: Scaling factor for updates.
    
    Returns:
        adapted_params: Dictionary of updated parameters.
    """
    adapted_params = clone_parameters(meta_model)
    for t in range(T):
        # Forward pass on support images for computing loss (optional).
        support_logits = meta_model(support_images)
        support_loss = F.cross_entropy(support_logits, support_labels)
        # Use the learnable projection layer to extract support features.
        support_feats = meta_model.extract_features(support_images)  # [L, projection_dim]
        updates = hypernet(support_feats)  # Dictionary: keys -> flat update vector
        
        new_adapted = {}
        for name, param in adapted_params.items():
            if name in updates:
                update_tensor = updates[name].view_as(param)
                new_adapted[name] = param + inner_lr * update_tensor
            else:
                new_adapted[name] = param
        adapted_params = new_adapted
    return adapted_params

def clone_parameters(model):
    """
    Clone model parameters into a dictionary.
    """
    return {name: param.clone() for name, param in model.named_parameters()}

###############################
# 5. Define the Transformer-based Hypernetwork
###############################
class Hypernetwork(nn.Module):
    """
    A hypernetwork that takes enhanced support features and outputs flat update vectors
    for selected model parameters (here, the classifier weights and bias).
    
    The enhanced support features are formed by concatenating:
      - The support embedding from the learnable projection.
      - The predicted logits from the meta_model for the support image.
      - The one-hot true label of the support image.
    """
    def __init__(self, input_dim, hidden_dim, output_dims_dict):
        super(Hypernetwork, self).__init__()
        # One transformer encoder layer.
        self.attn_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=4, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(self.attn_layer, num_layers=1)
        # MLP for each parameter we want to update.
        self.mlp_dict = nn.ModuleDict()
        for param_name, out_dim in output_dims_dict.items():
            self.mlp_dict[param_name] = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, out_dim)
            )
    
    def forward(self, enhanced_support_features):
        # enhanced_support_features: [L, input_dim]
        # Process the sequence with the transformer.
        seq = enhanced_support_features.unsqueeze(1)  # [L, 1, input_dim]
        attn_output = self.transformer_encoder(seq)     # [L, 1, input_dim]
        aggregated = attn_output.mean(dim=0).squeeze(0)   # [input_dim]
        # Generate update vectors for each model parameter.
        updates = {}
        for key, mlp in self.mlp_dict.items():
            update_vec = mlp(aggregated)
            updates[key] = update_vec
        return updates
###############################
# 6. Construct Meta-Learner and Optimizer
###############################
# num_classes = 5  # 5-way classification
# feature_dim = 64 * 5 * 5  # From BaseModel
# # We update only the classifier parameters.
# output_dims = {
#     'classifier_weight': num_classes * feature_dim,  # flattened weight matrix
#     'classifier_bias': num_classes
# }

# input_dim = 128   # Dimension for aggregated support features
# hidden_dim = 256

# hypernet = Hypernetwork(input_dim, hidden_dim, output_dims)
# Set dimensions.
projection_dim = 130   # For example
num_classes = 5        # 5-way classification
# New hypernetwork input dimension = projection_dim + 2*num_classes.
input_dim = projection_dim + 2 * num_classes
hidden_dim = 256

# Define the output_dims for the classifier parameters; they remain the same.
output_dims = {
    'classifier_weight': num_classes * (64 * 5 * 5),  # For example, if the classifier weight size is computed from feature dimensions.
    'classifier_bias': num_classes
}

# Instantiate the hypernetwork with the new input dimension.
hypernet = Hypernetwork(input_dim, hidden_dim, output_dims)

meta_model = BaseModel(num_classes=num_classes)

meta_optimizer = optim.Adam(list(meta_model.parameters()) + list(hypernet.parameters()), lr=1e-1)

###############################
# 7. Helper Function: Extract Support Features
###############################
def get_support_features(support_images, model):
    """
    Use the conv layers of the model to extract features from support images.
    Project the flattened features to input_dim using a simple linear projection.
    (For a real implementation, make this projection a learnable module.)
    
    Args:
        support_images: Tensor of shape [L, 3, 84, 84]
        model: The meta_model (we use its convolutional layers).
    
    Returns:
        features: Tensor of shape [L, input_dim].
    """
    with torch.no_grad():
        out = model.conv1(support_images)
        out = model.conv2(out)
        out = model.conv3(out)
        out = model.conv4(out)
        out = out.view(out.size(0), -1)  # [L, feature_dim]
    # Use a simple linear projection (here defined on the fly).
    proj = nn.Linear(out.size(1), input_dim).to(out.device)
    features = proj(out)
    return features

###############################
# 8. Updated Inner-Loop Adaptation (Support Set) with Enhanced Hypernetwork Input
###############################
def inner_loop_adaptation(meta_model, hypernet, support_images, support_labels, num_classes, T=3, inner_lr=1.0):
    """
    Adapt the meta_model to the current task using the support set.
    Updates only the classifier parameters using hypernetwork-generated updates.
    
    Args:
        meta_model: The universal model.
        hypernet: The hypernetwork.
        support_images: Tensor of support images.
        support_labels: Tensor of support labels.
        num_classes: Number of classes for one-hot encoding.
        T: Number of inner-loop iterations.
        inner_lr: Scaling factor for updates.
    
    Returns:
        adapted_params: Dictionary of updated parameters.
    """
    adapted_params = clone_parameters(meta_model)
    for t in range(T):
        # Optionally, compute the support loss (not used directly for the update here).
        support_logits = meta_model(support_images)
        support_loss = F.cross_entropy(support_logits, support_labels)
        
        # Extract the support features via the learnable projection.
        support_feats = meta_model.extract_features(support_images)  # shape: [L, projection_dim]
        # Get predicted logits for support images.
        support_logits = meta_model(support_images)  # shape: [L, num_classes]
        # Compute one-hot encoding for support labels.
        one_hot_labels = F.one_hot(support_labels, num_classes=num_classes).float()  # shape: [L, num_classes]
        # Concatenate along the feature dimension.
        enhanced_support_feats = torch.cat([support_feats, support_logits, one_hot_labels], dim=1)  # shape: [L, projection_dim + 2*num_classes]
        
        # Pass the enhanced features to the hypernetwork to generate updates.
        updates = hypernet(enhanced_support_feats)  # Dictionary mapping parameter names to update vectors.
        
        new_adapted = {}
        for name, param in adapted_params.items():
            if name in updates:
                update_tensor = updates[name].view_as(param)
                new_adapted[name] = param + inner_lr * update_tensor
            else:
                new_adapted[name] = param
        adapted_params = new_adapted
    return adapted_params


###############################
# 9. Outer-Loop Meta-Update (Query Set)
###############################
def meta_training_step(meta_model, hypernet, support_images, support_labels, query_images, query_labels, T=3, inner_lr=.3):
    """
    Performs one meta-training step:
      1. Inner-loop adaptation on the support set.
      2. Evaluate the adapted model on the query set.
    
    Returns:
        The query loss.
    """
    adapted_params = inner_loop_adaptation(meta_model, hypernet, support_images, support_labels, num_classes, T=T, inner_lr=inner_lr)
    
    # Temporarily update the meta_model classifier with adapted parameters.
    orig_weight = meta_model.classifier.weight.data.clone()
    orig_bias = meta_model.classifier.bias.data.clone()
    meta_model.classifier.weight.data.copy_(adapted_params["classifier.weight"].view_as(meta_model.classifier.weight))
    meta_model.classifier.bias.data.copy_(adapted_params["classifier.bias"].view_as(meta_model.classifier.bias))
    
    query_logits = meta_model(query_images)
    query_loss = F.cross_entropy(query_logits, query_labels)
    
    # Restore original parameters.
    meta_model.classifier.weight.data.copy_(orig_weight)
    meta_model.classifier.bias.data.copy_(orig_bias)
    
    return query_loss

###############################
# 10. Create a DataLoader for the CUB Meta Dataset (Training Split)
###############################
# Set the path to your CUB_200_2011 directory.
dataset_dir = "CUB_200_2011"  # <-- update this path
cub_meta_train_dataset = CubMetaDataset(dataset_dir=dataset_dir, split='train',
                                        num_classes=5, num_support=5, num_query=15,
                                        transform=transform, train_ratio=0.8)
train_dataloader = DataLoader(cub_meta_train_dataset, batch_size=1, shuffle=True)

###############################
# 11. Meta-Training Loop
###############################
num_meta_iterations = 500  # Adjust as needed

meta_model.train()
for iteration, episode in enumerate(train_dataloader):
    if iteration >= num_meta_iterations:
        break
    # Each episode is a tuple: (support_images, support_labels, query_images, query_labels)
    support_images, support_labels, query_images, query_labels = episode
    # Remove extra batch dimension (since batch_size=1)
    support_images = support_images.squeeze(0)
    support_labels = support_labels.squeeze(0)
    query_images = query_images.squeeze(0)
    query_labels = query_labels.squeeze(0)
    
    meta_optimizer.zero_grad()
    loss = meta_training_step(meta_model, hypernet,
                              support_images, support_labels,
                              query_images, query_labels,
                              T=3, inner_lr=1)
    loss.backward()
    meta_optimizer.step()
    
    if iteration % 50 == 0:
        print(f"Iteration {iteration}: Meta-training loss = {loss.item():.4f}")

###############################
# 12. (Optional) Meta-Testing and Plotting Results
###############################
def evaluate_meta_model(meta_model, hypernet, dataset, T=3, inner_lr=1.0, num_episodes=100):
    """
    Evaluate the meta-model on a set of episodes from the test split.
    Returns average test accuracy and also saves the first episode's query images,
    predicted labels, and ground truth labels for plotting.
    """
    meta_model.eval()
    total_correct = 0
    total_samples = 0
    first_episode_images = None
    first_episode_preds = None
    first_episode_labels = None

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    with torch.no_grad():
        for i, episode in enumerate(dataloader):
            if i >= num_episodes:
                break
            support_images, support_labels, query_images, query_labels = episode
            support_images = support_images.squeeze(0)
            support_labels = support_labels.squeeze(0)
            query_images = query_images.squeeze(0)
            query_labels = query_labels.squeeze(0)
            
            adapted_params = inner_loop_adaptation(meta_model, hypernet, support_images, support_labels,num_classes=5, T=T, inner_lr=inner_lr)
            orig_weight = meta_model.classifier.weight.data.clone()
            orig_bias = meta_model.classifier.bias.data.clone()
            meta_model.classifier.weight.data.copy_(adapted_params["classifier.weight"].view_as(meta_model.classifier.weight))
            meta_model.classifier.bias.data.copy_(adapted_params["classifier.bias"].view_as(meta_model.classifier.bias))
            
            query_logits = meta_model(query_images)
            preds = torch.argmax(query_logits, dim=1)
            total_correct += (preds == query_labels).sum().item()
            total_samples += query_labels.size(0)
            
            if i == 0:
                first_episode_images = query_images
                first_episode_preds = preds
                first_episode_labels = query_labels

            meta_model.classifier.weight.data.copy_(orig_weight)
            meta_model.classifier.bias.data.copy_(orig_bias)
    
    avg_acc = total_correct / total_samples * 100.0
    print(f"Test Accuracy over {num_episodes} episodes: {avg_acc:.2f}%")
    return avg_acc, first_episode_images, first_episode_preds, first_episode_labels

# Create a test dataset (using split='test')
cub_meta_test_dataset = CubMetaDataset(dataset_dir=dataset_dir, split='test',
                                       num_classes=5, num_support=5, num_query=5,
                                       transform=transform, train_ratio=0.8)
test_acc, first_q_images, first_q_preds, first_q_labels = evaluate_meta_model(meta_model, hypernet,
                                                                              cub_meta_test_dataset,
                                                                              T=3, inner_lr=1.0,
                                                                              num_episodes=100)

###############################
# 13. Plot the Results of the First Test Episode
###############################
def plot_episode_results(images, preds, labels):
    """
    Plot query images with ground-truth and predicted labels.
    
    Args:
        images: Tensor of shape [num_query, 3, 84,84].
        preds: Tensor of predicted labels.
        labels: Tensor of ground-truth labels.
    """
    images = images.cpu()
    num_images = images.size(0)
    plt.figure(figsize=(15, 3))
    for i in range(num_images):
        plt.subplot(1, num_images, i+1)
        img = images[i].permute(1, 2, 0).numpy()
        # Reverse normalization.
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        img = img * std + mean
        img = img.clip(0, 1)
        plt.imshow(img)
        plt.title(f"GT: {labels[i].item()}\nPred: {preds[i].item()}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Plot the query set of the first test episode.
plot_episode_results(first_q_images, first_q_preds, first_q_labels)
