import numpy as np
import torch
import torch.nn.functional as F

def compute_accuracy(predictions, targets):
    """
    Computes the accuracy of the model predictions.
    
    Args:
        predictions (torch.Tensor): Model predictions (logits) [batch_size, num_classes]
        targets (torch.Tensor): Ground truth [batch_size]
        
    Returns:
        float: Accuracy between 0 y 1
    """
    if predictions.shape[1] > 1:  #If multiclass classification
        pred_classes = predictions.argmax(dim=1)
    else: #If binary classification
        pred_classes = (predictions > 0).long().view(-1)
    
    return (pred_classes == targets).float().mean().item()

def compute_confidence_metrics(logits, targets):
    """
    Computes various confidence metrics from model predictions.
    
    Args:
        logits (torch.Tensor): Model predictions [batch_size, num_classes]
        targets (torch.Tensor): Ground Truth [batch_size]
        
    Returns:
        dict: Metrics including accuracy, mean confidence, confidence for correct and incorrect predictions
    """

    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits)
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets)

    if logits.dim() == 1:
        logits = logits.unsqueeze(1)

    probs = F.softmax(logits, dim=1)
    
    pred_classes = probs.argmax(dim=1)
    confidences = probs.max(dim=1)[0]
    
    correct = (pred_classes == targets)
    
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
    Computes the Expected Calibration Error (ECE).
    
    Args:
        confidences (np.ndarray): Confidence scores [batch_size]
        correctness (np.ndarray): Binary array [batch_size] indicating if the prediction was correct
        n_bins (int, optional): Number of bins to use for calibration (default: 10)
        
    Returns:
        tuple: (ECE, bins_data) where the bins_data is a dictionary containing:
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidences, bin_edges[1:-1])
    
    bin_accs = np.zeros(n_bins)    # Accuracy
    bin_confs = np.zeros(n_bins)   # Confidence
    bin_counts = np.zeros(n_bins)  # Number of samples in each bin
    
    for i in range(len(confidences)):
        bin_idx = bin_indices[i]
        bin_accs[bin_idx] += correctness[i]
        bin_confs[bin_idx] += confidences[i]
        bin_counts[bin_idx] += 1
    
    valid_bins = bin_counts > 0
    bin_accs[valid_bins] /= bin_counts[valid_bins]
    bin_confs[valid_bins] /= bin_counts[valid_bins]
    
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
    Computes per-class accuracy and count of samples for each class.
    
    Args:
        logits (torch.Tensor): Model logits [batch_size, n_classes]
        targets (torch.Tensor): Ground truth [batch_size]
        n_classes (int): Number of classes
        
    Returns:
        dict: Metric per class including accuracy and count
    """
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits)
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets)
    
    probs = F.softmax(logits, dim=1)
    preds = probs.argmax(dim=1)
    
    per_class_acc = []
    per_class_count = []
    
    for c in range(n_classes):
        class_mask = (targets == c)
        class_count = class_mask.sum().item()
        
        if class_count > 0:
            class_acc = (preds[class_mask] == targets[class_mask]).float().mean().item()
        else:
            class_acc = 0.0
        
        per_class_acc.append(class_acc)
        per_class_count.append(class_count)
    
    return {
        'per_class_acc': per_class_acc,
        'per_class_count': per_class_count
    }