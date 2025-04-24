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
from tqdm import tqdm
###############################
# (1) Dataset Code (your code)
###############################
def split_classes(all_labels, train_ratio=0.8):
    unique_classes = list(set(all_labels))
    random.shuffle(unique_classes)
    num_train = int(len(unique_classes) * train_ratio)
    train_classes = unique_classes[:num_train]
    test_classes = unique_classes[num_train:]
    return train_classes, test_classes

class CubMetaDataset(Dataset):
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

        self.image_names = []
        with open(self.images_file, "r") as f:
            for line in f:
                _, img_filename = line.strip().split()
                self.image_names.append(img_filename)

        self.labels = []
        with open(self.labels_file, "r") as f:
            for line in f:
                _, label = line.strip().split()
                self.labels.append(int(label))
        
        assert len(self.image_names) == len(self.labels), "Mismatch between images and labels."
        self.train_classes, self.test_classes = split_classes(self.labels, train_ratio=train_ratio)
        if split == 'train':
            self.classes = self.train_classes
        else:
            self.classes = self.test_classes
        
        self.class_to_indices = {cls: [] for cls in self.classes}
        for idx, label in enumerate(self.labels):
            if label in self.classes:
                self.class_to_indices[label].append(idx)
        
        self.episode_length = 100000

    def __len__(self):
        return self.episode_length

    def __getitem__(self, index):
        chosen_classes = random.sample(self.classes, self.num_classes)
        support_set = []
        query_set = []
        support_labels = []
        query_labels = []
        for new_label, cls in enumerate(chosen_classes):
            indices = self.class_to_indices[cls]
            if len(indices) < self.num_support + self.num_query:
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
                support_set.append(img)
                support_labels.append(new_label)
            
            for i in query_indices:
                img_path = os.path.join(self.image_dir, self.image_names[i])
                img = Image.open(img_path).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                query_set.append(img)
                query_labels.append(new_label)
        support_images = torch.stack(support_set, dim=0)
        query_images = torch.stack(query_set, dim=0)
        support_labels = torch.tensor(support_labels)
        query_labels = torch.tensor(query_labels)
        return support_images, support_labels, query_images, query_labels

###############################
# (2) Model Components
###############################

# Feature extractor backbone with uncertainty outputs
class FeatureExtractor(nn.Module):
    def __init__(self, output_dim=256):
        super(FeatureExtractor, self).__init__()
        # A simple 3–layer CNN. In practice, you might use a pre-trained network.
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # [B, 64, H, W]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 64, H/2, W/2]
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # [B, 128, H/2, W/2]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 128, H/4, W/4]
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # [B, 256, H/4, W/4]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # [B, 256, 1, 1]
        )
        self.fc_mean = nn.Linear(256, output_dim)
        self.fc_logvar = nn.Linear(256, output_dim)
        
    def forward(self, x):
        feat = self.conv(x).view(x.size(0), -1)
        mean = self.fc_mean(feat)
        logvar = self.fc_logvar(feat)
        # For simplicity, return the mean as embedding.
        return mean, logvar

# Transformer-based support aggregator.
class SupportAggregator(nn.Module):
    def __init__(self, emb_dim, num_classes, num_layers=2, num_heads=4):
        super(SupportAggregator, self).__init__()
        self.num_classes = num_classes
        # Learned class tokens (one per class).
        self.class_tokens = nn.Parameter(torch.randn(num_classes, emb_dim))
        # Embedding for segment (0 for class token, 1 for support token).
        self.segment_embed = nn.Embedding(2, emb_dim)
        # Embedding for the class id (used to augment support tokens).
        self.class_id_embed = nn.Embedding(num_classes, emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, support_embeddings, support_labels):
        # support_embeddings: [batch, num_support_total, emb_dim]
        # support_labels: [batch, num_support_total] with values 0,...,num_classes-1
        batch_size = support_embeddings.size(0)
        device = support_embeddings.device
        # Replicate learned class tokens for each task.
        class_tokens = self.class_tokens.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, num_classes, emb_dim]
        
        # For support tokens, add an embedding for their class id.
        support_class_embed = self.class_id_embed(support_labels)  # [batch, num_support_total, emb_dim]
        support_embeddings = support_embeddings + support_class_embed
        
        # Add segment embeddings: use 0 for class tokens, 1 for support tokens.
        seg_class = self.segment_embed(torch.zeros((batch_size, self.num_classes), dtype=torch.long, device=device))
        seg_support = self.segment_embed(torch.ones((batch_size, support_embeddings.size(1)), dtype=torch.long, device=device))
        class_tokens = class_tokens + seg_class
        support_embeddings = support_embeddings + seg_support
        
        # Concatenate class tokens and support tokens along the sequence dimension.
        tokens = torch.cat([class_tokens, support_embeddings], dim=1)  # [batch, num_classes + num_support, emb_dim]
        # Transformer expects (seq_len, batch, emb_dim)
        tokens = tokens.transpose(0, 1)
        refined_tokens = self.transformer(tokens)
        # Extract refined class tokens (first num_classes tokens)
        refined_prototypes = refined_tokens[:self.num_classes].transpose(0, 1)  # [batch, num_classes, emb_dim]
        return refined_prototypes

# Hypernetwork that generates classifier weights for each class.
class Hypernetwork(nn.Module):
    def __init__(self, proto_dim, out_dim):
        super(Hypernetwork, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(proto_dim, proto_dim),
            nn.ReLU(),
            nn.Linear(proto_dim, out_dim + 1)  # out_dim for weight, 1 for bias.
        )
    def forward(self, prototypes):
        # prototypes: [batch, num_classes, proto_dim]
        out = self.mlp(prototypes)  # [batch, num_classes, out_dim+1]
        weights = out[:, :, :-1]   # [batch, num_classes, out_dim]
        bias = out[:, :, -1]       # [batch, num_classes]
        return weights, bias

# The overall meta-learner.
class MetaLearner(nn.Module):
    def __init__(self, emb_dim=256, num_layers=2, num_heads=4, num_classes=5, num_support=1):
        super(MetaLearner, self).__init__()
        self.emb_dim = emb_dim
        self.num_classes = num_classes
        self.num_support = num_support
        # Backbone feature extractor.
        self.feature_extractor = FeatureExtractor(output_dim=emb_dim)
        # Transformer-based support aggregator.
        self.support_aggregator = SupportAggregator(emb_dim=emb_dim, num_classes=num_classes,
                                                    num_layers=num_layers, num_heads=num_heads)
        # Hypernetwork to generate classifier parameters.
        self.hypernetwork = Hypernetwork(proto_dim=emb_dim, out_dim=emb_dim)
    
    def forward(self, support_images, support_labels, query_images):
        # support_images: [batch, num_support_total, C, H, W]
        # query_images: [batch, num_query_total, C, H, W]
        batch_size = support_images.size(0)
        # Process support images.
        support_images = support_images.view(-1, *support_images.shape[2:])  # flatten support tokens
        support_mean, support_logvar = self.feature_extractor(support_images)
        support_feat = support_mean  # using mean as representation
        support_feat = support_feat.view(batch_size, -1, self.emb_dim)  # [batch, num_support_total, emb_dim]

        # Use support aggregator to get refined prototypes.
        prototypes = self.support_aggregator(support_feat, support_labels)  # [batch, num_classes, emb_dim]
        
        # Generate classifier parameters from prototypes.
        weights, bias = self.hypernetwork(prototypes)  # weights: [batch, num_classes, emb_dim], bias: [batch, num_classes]

        # Process query images.
        query_images = query_images.view(-1, *query_images.shape[2:])
        query_mean, query_logvar = self.feature_extractor(query_images)
        query_feat = query_mean  # [batch * num_query, emb_dim]
        query_feat = query_feat.view(batch_size, -1, self.emb_dim)  # [batch, num_query_total, emb_dim]

        # For each task in the batch, compute logits for query examples.
        logits = []
        for i in range(batch_size):
            # For task i: query_feat[i]: [num_query, emb_dim]
            # weights[i]: [num_classes, emb_dim] -> classifier weights.
            # Compute linear logits: dot product plus bias.
            logit = query_feat[i] @ weights[i].t() + bias[i].unsqueeze(0)
            logits.append(logit)
        logits = torch.stack(logits, dim=0)  # [batch, num_query, num_classes]
        return logits

###############################
# (3) Training & Evaluation Loops
###############################
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    for support_images, support_labels, query_images, query_labels in tqdm(dataloader, desc="Training Epoch"):
        support_images = support_images.to(device)
        support_labels = support_labels.to(device)
        query_images = query_images.to(device)
        query_labels = query_labels.to(device)
        optimizer.zero_grad()
        logits = model(support_images, support_labels, query_images)  # [batch, num_query, num_classes]
        # Flatten logits and labels for cross-entropy.
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), query_labels.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for support_images, support_labels, query_images, query_labels in dataloader:
            support_images = support_images.to(device)
            support_labels = support_labels.to(device)
            query_images = query_images.to(device)
            query_labels = query_labels.to(device)
            logits = model(support_images, support_labels, query_images)  # [batch, num_query, num_classes]
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == query_labels).sum().item()
            total += query_labels.numel()
    return correct / total

###############################
# (4) Putting It All Together
###############################

if __name__ == '__main__':
    # Hyperparameters.
    emb_dim = 256
    num_classes = 5
    num_support = 1
    num_query = 5
    transformer_layers = 2
    transformer_heads = 4
    batch_size = 4  # Number of episodes per batch.
    num_epochs = 20
    learning_rate = 1e-3

    # Set device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data transforms (adjust size as needed for CUB images).
    transform = transforms.Compose([
        transforms.Resize((84, 84)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    
    # Directory for CUB dataset (update path as required).
    dataset_dir = 'CUB_200_2011'
    
    # Create train and test datasets.
    train_dataset = CubMetaDataset(dataset_dir=dataset_dir, split='train', 
                                   num_classes=num_classes, num_support=num_support,
                                   num_query=num_query, transform=transform)
    test_dataset = CubMetaDataset(dataset_dir=dataset_dir, split='test', 
                                  num_classes=num_classes, num_support=num_support,
                                  num_query=num_query, transform=transform)
    
    # Data loaders.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Initialize model.
    model = MetaLearner(emb_dim=emb_dim, num_layers=transformer_layers,
                        num_heads=transformer_heads, num_classes=num_classes,
                        num_support=num_support)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop.
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        test_acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f} - Test Acc: {test_acc*100:.2f}%")
    
    # Save the model.
    torch.save(model.state_dict(), "meta_learner_cub.pth")
