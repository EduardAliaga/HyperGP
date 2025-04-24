import torch
import torch.nn as nn

class ConvEmbedder(nn.Module):
    """
    Convolutional neural network embedder for few-shot learning.
    Standard architecture with 4 convolutional blocks followed by average pooling.
    """
    def __init__(self, out_dim, n_layers):
        """
        Args:
            out_dim (int): Output dimension of the embedder (feature dimension)
            n_layers (int): Number of convolutional blocks
        """
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
        """
        Forward pass through the embedder.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 3, H, W]
            
        Returns:
            torch.Tensor: Embedded features of shape [batch_size, out_dim]
        """
        features = self.encoder(x)
        return features.view(x.size(0), -1)


class Classifier(nn.Module):
    """
    Simple linear classifier for pre-training the embedder.
    """
    def __init__(self, in_features, num_classes):
        """
        Args:
            in_features (int): Input feature dimension
            num_classes (int): Number of output classes
        """
        super().__init__()
        self.classifier = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        """
        Forward pass through the classifier.
        
        Args:
            x (torch.Tensor): Input features of shape [batch_size, in_features]
            
        Returns:
            torch.Tensor: Logits of shape [batch_size, num_classes]
        """
        return self.classifier(x)