import torch
import torch.nn as nn
import torch.nn.functional as F

class HyperNet(nn.Module):
    """
    Hypernetwork that generates Gaussian process kernel parameters.
    Takes support set embeddings and one-hot encoded labels as input.
    Outputs kernel hyperparameters: lengthscales, signal amplitude, and noise level.
    """
    def __init__(self, NK, D, H, n_hidden, N_way):
        """
        Args:
            NK (int): Number of support examples (N_way * K_shot)
            D (int): Dimension of the feature embeddings
            H (int): Hidden dimension of the hypernetwork
            n_hidden (int): Number of hidden layers
            N_way (int): Number of classes in the task
        """
        super().__init__()
        # Capas de la red - Usando LayerNorm para estabilidad con batch=1
        layers = [nn.Linear(NK*(D+N_way), H), nn.LayerNorm(H), nn.ReLU(inplace=True)]
        for _ in range(n_hidden-1):
            layers += [nn.Linear(H, H), nn.LayerNorm(H), nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*layers)
        
        # Capas de salida para los hiperparámetros
        self.log_ell = nn.Linear(H, D)
        self.log_sf = nn.Linear(H, 1)
        self.log_sn = nn.Linear(H, 1)
        
        # Inicialización cuidadosa
        nn.init.normal_(self.log_ell.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.log_ell.bias, 0.0)  # Inicializa para log_ell ≈ 1.0
        
        nn.init.normal_(self.log_sf.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.log_sf.bias, 0.0)   # Inicializa para log_sf ≈ 1.0
        
        nn.init.normal_(self.log_sn.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.log_sn.bias, -3.0)  # Mantén el valor bajo para sn
        
    def forward(self, feats, labels_onehot):
        """
        Forward pass through the hypernetwork.
        
        Args:
            feats (torch.Tensor): Support set feature embeddings [NK, D]
            labels_onehot (torch.Tensor): One-hot encoded support labels [NK, N_way]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                - ell: lengthscales per dimension [D]
                - sf: signal amplitude (scalar)
                - sn: noise level (scalar)
        """
        # Normalización L2 de características para estabilidad
        feats_norm = F.normalize(feats, p=2, dim=1)
        
        # Concatenación y procesamiento
        inp = torch.cat([feats_norm, labels_onehot], dim=1).view(1, -1)
        h = self.net(inp)
        
        # Generación de hiperparámetros con límites suavizados
        ell = 0.1 + 10.0 * torch.sigmoid(self.log_ell(h))  # [0.1, 10.1]
        sf = 0.5 + 2.0 * torch.sigmoid(self.log_sf(h))     # [0.5, 2.5]
        sn = 1e-3 + 0.1 * torch.sigmoid(self.log_sn(h))    # [0.001, 0.101]
        
        return ell.view(-1), sf.view(-1), sn.view(-1)