import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
import seaborn as sns

def save_task_example(sx, sy, qx, qy, preds, output_path):
    """
    Saves a task example with support and query images.
    
    Args:
        sx: Support images [N_way*K_shot, 3, H, W]
        sy: Support labels [N_way*K_shot]
        qx: Query images [N_way*Q_query, 3, H, W]
        qy: Query labels [N_way*Q_query]
        preds: Query predictions 
        output_path: save path
    """

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    n_support = sx.size(0)
    n_query = min(10, qx.size(0))  
    
    plt.figure(figsize=(15, 8))

    for i in range(n_support):
        plt.subplot(2, max(n_support, n_query), i+1)
        img = sx[i].cpu()
        img = img * std + mean
        img = img.permute(1, 2, 0).clamp(0, 1).numpy()
        plt.imshow(img)
        plt.title(f"Support: Class {sy[i].item()}")
        plt.axis('off')
    

    for i in range(n_query):
        plt.subplot(2, max(n_support, n_query), i+1+max(n_support, n_query))
        img = qx[i].cpu()
        img = img * std + mean
        img = img.permute(1, 2, 0).clamp(0, 1).numpy()
        plt.imshow(img)
        correct = (preds[i] == qy[i].item())
        plt.title(f"Query: True {qy[i].item()}, Pred {preds[i]}", 
                 color=('green' if correct else 'red'))
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    return output_path

def plot_learning_curves(train_losses, val_epochs, val_accs, output_path):
    """
    Generates a learning curve visualization.
    
    Args:
        train_losses: epoch losses
        val_epochs: validation epochs losses
        val_accs: validation accuracies
        output_path: save path
    """
    plt.figure(figsize=(15, 5))
    
    # Curva de pérdida de entrenamiento
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(1, len(train_losses)+1), train_losses)
    plt.title('Training Loss')
    plt.xlabel('Episodes')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Curva de precisión de validación
    plt.subplot(1, 2, 2)
    plt.plot(val_epochs, val_accs, marker='o')
    plt.title('Validation Accuracy')
    plt.xlabel('Episodes')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    return output_path

def plot_kernel_params(kernel_params, gradient_norms_hypernet=None, gradient_norms_embedder=None, output_path=None):
    """
    Kernel parameters visualization.
    
    Args:
        kernel_params: list of dictionaries with kernel parameters
        gradient_norms_hypernet: gradient norms of hypernetwork
        gradient_norms_embedder: gradient norms of embedder
        output_path: save path
    """
    if not kernel_params:
        print("The kernel_params list is empty.")
        return None

    epochs = [item['epoch'] for item in kernel_params]
    ell_mins = [item['ell_min'] for item in kernel_params]
    ell_means = [item['ell_mean'] for item in kernel_params]
    ell_maxs = [item['ell_max'] for item in kernel_params]
    sfs = [item['sf'] for item in kernel_params]
    sns = [item['sn'] for item in kernel_params]
    
    plt.figure(figsize=(15, 10))
 
    plt.subplot(2, 2, 1)
    plt.plot(epochs, ell_mins, label='Min')
    plt.plot(epochs, ell_means, label='Mean')
    plt.plot(epochs, ell_maxs, label='Max')
    plt.title('Lengthscale (ell) Evolution')
    plt.xlabel('Episodes')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(epochs, sfs)
    plt.title('Signal Amplitude (sf) Evolution')
    plt.xlabel('Episodes')
    plt.ylabel('Value')
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(epochs, sns)
    plt.title('Noise Level (sn) Evolution')
    plt.xlabel('Episodes')
    plt.ylabel('Value')
    plt.grid(True)

    plt.subplot(2, 2, 4)

    if gradient_norms_hypernet:
        if isinstance(gradient_norms_hypernet[0], dict):
            hyp_epochs = [item['epoch'] for item in gradient_norms_hypernet]
            hyp_values = [item['norm'] for item in gradient_norms_hypernet]
        else:
            hyp_epochs = epochs
            hyp_values = gradient_norms_hypernet
        
        plt.plot(hyp_epochs, hyp_values, label='HyperNet')
    
    if gradient_norms_embedder:
        if isinstance(gradient_norms_embedder[0], dict):
            emb_epochs = [item['epoch'] for item in gradient_norms_embedder]
            emb_values = [item['norm'] for item in gradient_norms_embedder]
        else:
            emb_epochs = epochs
            emb_values = gradient_norms_embedder
            
        plt.plot(emb_epochs, emb_values, label='Embedder')
    
    plt.title('Gradient Norm Evolution')
    plt.xlabel('Episodes')
    plt.ylabel('Norm')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        plt.close()
        return output_path
    else:
        plt.show()
        plt.close()
        return None

def plot_accuracy_distribution(test_accs, output_path=None):
    """
    Accuracy distribution visualization.
    
    Args:
        test_accs: accuracies of test tasks
        output_path: save_path
    """
    mean_acc = np.mean(test_accs)
    median_acc = np.median(test_accs)
    std_acc = np.std(test_accs)
    ci95 = 1.96 * std_acc / np.sqrt(len(test_accs))
    
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(test_accs, bins=20, alpha=0.7, color='#1f77b4', edgecolor='black')
    plt.axvline(mean_acc, color='red', linestyle='dashed', linewidth=2, 
                label=f'Mean: {mean_acc*100:.2f}%')
    plt.axvline(median_acc, color='green', linestyle='dashed', linewidth=2, 
                label=f'Median: {median_acc*100:.2f}%')
    plt.title(f'Accuracy Distribution ({len(test_accs)} tasks)')
    plt.xlabel('Accuracy')
    plt.ylabel('Number of Tasks')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.boxplot(test_accs, vert=True, patch_artist=True)
    plt.title(f'Accuracy: {mean_acc*100:.2f}% ± {ci95*100:.2f}%')
    plt.ylabel('Accuracy')
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        plt.close()
        return output_path
    else:
        plt.show()
        plt.close()
        return None

def plot_calibration(confidences, correctness, bins=10, output_path=None):
    """
    Genera un diagrama de fiabilidad (calibration plot).
    
    Args:
        confidences (np.ndarray): Array de confidencias de predicción
        correctness (np.ndarray): Array de 1s y 0s indicando si la predicción fue correcta
        bins (int, optional): Número de bins para el análisis
        output_path (str, optional): Ruta para guardar la visualización
        
    Returns:
        tuple: (ECE, ruta del archivo guardado)
    """
    # Crear bins de confidencia
    bin_edges = np.linspace(0, 1, bins + 1)
    bin_indices = np.digitize(confidences, bin_edges[1:-1])
    
    # Calcular precisión y confianza promedio por bin
    bin_accs = np.zeros(bins)
    bin_confs = np.zeros(bins)
    bin_counts = np.zeros(bins)
    
    for i in range(len(confidences)):
        bin_idx = bin_indices[i]
        bin_accs[bin_idx] += correctness[i]
        bin_confs[bin_idx] += confidences[i]
        bin_counts[bin_idx] += 1
    
    # Calcular promedios (evitar división por cero)
    for i in range(bins):
        if bin_counts[i] > 0:
            bin_accs[i] /= bin_counts[i]
            bin_confs[i] /= bin_counts[i]
    
    # Calcular Expected Calibration Error (ECE)
    ece = np.sum(np.abs(bin_accs - bin_confs) * (bin_counts / len(confidences)))
    
    # Diagrama de fiabilidad (Calibration Plot)
    plt.figure(figsize=(10, 10))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    plt.bar(bin_edges[:-1], bin_accs, width=1/bins, alpha=0.8, align='edge', 
            edgecolor='black', color='#1f77b4', label='Accuracy in bin')
    plt.plot(bin_confs, bin_accs, 'ro-', label='Accuracy vs Confidence')
    
    plt.title(f'Calibration Plot (ECE: {ece:.4f})')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        plt.close()
        return ece, output_path
    else:
        plt.show()
        plt.close()
        return ece, None

def visualize_kernel_weights(ell, input_dim=2, output_path=None):
    """
    Visualiza los pesos de lengthscale del kernel para entender qué características
    son importantes para la tarea.
    
    Args:
        ell (torch.Tensor): Lengthscales por dimensión [D]
        input_dim (int, optional): Dimensionalidad original de entrada para reshaping
        output_path (str, optional): Ruta para guardar la visualización
        
    Returns:
        str: Ruta del archivo guardado o None
    """
    # Las lengthscales más pequeñas corresponden a dimensiones más importantes
    importance = 1.0 / ell.cpu().numpy()
    
    # Si la dimensión del embedding es un múltiplo de input_dim, 
    # podemos visualizar como un heatmap 2D
    if len(importance) % input_dim == 0:
        channels = len(importance) // input_dim
        img = importance.reshape(channels, input_dim, input_dim)
        
        plt.figure(figsize=(12, 8))
        for i in range(min(channels, 16)):  # Mostrar máx. 16 canales
            plt.subplot(4, 4, i+1)
            plt.imshow(img[i], cmap='viridis')
            plt.colorbar()
            plt.title(f'Channel {i+1}')
            plt.axis('off')
    else:
        # Si no, simplemente mostramos como un heatmap unidimensional
        plt.figure(figsize=(12, 6))
        plt.imshow(importance.reshape(1, -1), cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.title('Feature Importance Based on Kernel Lengthscales')
        plt.xlabel('Feature Dimension')
        
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        plt.close()
        return output_path
    else:
        plt.show()
        plt.close()
        return None