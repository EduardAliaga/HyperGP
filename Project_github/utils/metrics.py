import numpy as np
import torch
import torch.nn.functional as F

def compute_accuracy(predictions, targets):
    """
    Calcula la precisión de clasificación.
    
    Args:
        predictions (torch.Tensor): Predicciones del modelo (logits) [batch_size, num_classes]
        targets (torch.Tensor): Etiquetas verdaderas [batch_size]
        
    Returns:
        float: Precisión entre 0 y 1
    """
    if predictions.shape[1] > 1:  # Si son logits multi-clase
        pred_classes = predictions.argmax(dim=1)
    else:  # Si son logits binarios
        pred_classes = (predictions > 0).long().view(-1)
    
    return (pred_classes == targets).float().mean().item()

def compute_confidence_metrics(logits, targets):
    """
    Calcula métricas relacionadas con la confianza de las predicciones.
    
    Args:
        logits (torch.Tensor): Predicciones del modelo [batch_size, num_classes]
        targets (torch.Tensor): Etiquetas verdaderas [batch_size]
        
    Returns:
        dict: Diccionario con varias métricas
    """
    # Convertir a tensores PyTorch si no lo son
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits)
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets)
    
    # Asegurarse de que estén en la misma dimensión
    if logits.dim() == 1:
        logits = logits.unsqueeze(1)
    
    # Calcular probabilidades
    probs = F.softmax(logits, dim=1)
    
    # Predicciones y confianza
    pred_classes = probs.argmax(dim=1)
    confidences = probs.max(dim=1)[0]
    
    # Calcular corrección
    correct = (pred_classes == targets)
    
    # Métricas
    accuracy = correct.float().mean().item()
    mean_confidence = confidences.mean().item()
    confidence_correct = confidences[correct].mean().item() if correct.sum() > 0 else 0.0
    confidence_incorrect = confidences[~correct].mean().item() if (~correct).sum() > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'mean_confidence': mean_confidence,
        'confidence_correct': confidence_correct,
        'confidence_incorrect': confidence_incorrect
    }

def expected_calibration_error(confidences, correctness, n_bins=10):
    """
    Calcula el Expected Calibration Error (ECE).
    
    Args:
        confidences (np.ndarray): Array de confidencias de predicción
        correctness (np.ndarray): Array de 1s y 0s indicando si la predicción fue correcta
        n_bins (int, optional): Número de bins para el análisis
        
    Returns:
        tuple: (ECE, bins_data) donde bins_data contiene estadísticas por bin
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidences, bin_edges[1:-1])
    
    bin_accs = np.zeros(n_bins)    # Precisión promedio por bin
    bin_confs = np.zeros(n_bins)   # Confianza promedio por bin
    bin_counts = np.zeros(n_bins)  # Número de muestras por bin
    
    for i in range(len(confidences)):
        bin_idx = bin_indices[i]
        bin_accs[bin_idx] += correctness[i]
        bin_confs[bin_idx] += confidences[i]
        bin_counts[bin_idx] += 1
    
    # Calcular promedios (evitar división por cero)
    valid_bins = bin_counts > 0
    bin_accs[valid_bins] /= bin_counts[valid_bins]
    bin_confs[valid_bins] /= bin_counts[valid_bins]
    
    # Calcular ECE
    ece = np.sum(np.abs(bin_accs - bin_confs) * (bin_counts / len(confidences)))
    
    bins_data = {
        'bin_edges': bin_edges,
        'bin_accs': bin_accs,
        'bin_confs': bin_confs,
        'bin_counts': bin_counts
    }
    
    return ece, bins_data

def compute_per_class_metrics(logits, targets, n_classes):
    """
    Calcula métricas desagregadas por clase.
    
    Args:
        logits (torch.Tensor): Logits del modelo [batch_size, n_classes]
        targets (torch.Tensor): Etiquetas verdaderas [batch_size]
        n_classes (int): Número de clases
        
    Returns:
        dict: Métricas por clase
    """
    # Convertir a tensores PyTorch si no lo son
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits)
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets)
    
    probs = F.softmax(logits, dim=1)
    preds = probs.argmax(dim=1)
    
    per_class_acc = []
    per_class_count = []
    
    for c in range(n_classes):
        # Ejemplos de esta clase
        class_mask = (targets == c)
        class_count = class_mask.sum().item()
        
        if class_count > 0:
            # Precisión para esta clase
            class_acc = (preds[class_mask] == targets[class_mask]).float().mean().item()
        else:
            class_acc = 0.0
        
        per_class_acc.append(class_acc)
        per_class_count.append(class_count)
    
    return {
        'per_class_acc': per_class_acc,
        'per_class_count': per_class_count
    }