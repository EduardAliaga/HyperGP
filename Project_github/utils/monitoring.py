import numpy as np
import torch
from tqdm import tqdm
import time
import os

def monitor_params(ep, ell, sf, sn, hypernet, embedder=None, finetune_embedder=False):
    """
    Monitorea los parámetros del kernel y las normas de gradientes.
    
    Args:
        ep (int): Época actual
        ell (torch.Tensor): Lengthscales del kernel
        sf (torch.Tensor): Amplitud de señal del kernel
        sn (torch.Tensor): Nivel de ruido del kernel
        hypernet (nn.Module): Modelo de hypernetwork
        embedder (nn.Module, optional): Modelo embedder
        finetune_embedder (bool, optional): Si el embedder se está fine-tuneando
    
    Returns:
        dict: Estadísticas de parámetros y gradientes
    """
    print(f"\nEpoch {ep} monitoring:")
    print(f"  ell range: min={ell.min().item():.4f}, mean={ell.mean().item():.4f}, max={ell.max().item():.4f}")
    print(f"  sf: {sf.item():.4f}, sn: {sn.item():.4f}")
    
    stats = {
        'epoch': ep,
        'ell_min': ell.min().item(),
        'ell_mean': ell.mean().item(),
        'ell_max': ell.max().item(),
        'sf': sf.item(),
        'sn': sn.item(),
        'hypernet_grad_norm': 0.0,
        'embedder_grad_norm': 0.0
    }
    
    # Norma de gradientes de hypernet
    hypernet_grad_norm = 0.0
    for p in hypernet.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            hypernet_grad_norm += param_norm.item() ** 2
    hypernet_grad_norm = hypernet_grad_norm ** 0.5
    stats['hypernet_grad_norm'] = hypernet_grad_norm
    print(f"  HyperNet Gradient norm: {hypernet_grad_norm:.4f}")
    
    # Norma de gradientes del embedder (si está siendo entrenado)
    if finetune_embedder and embedder is not None:
        embedder_grad_norm = 0.0
        for p in embedder.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                embedder_grad_norm += param_norm.item() ** 2
        embedder_grad_norm = embedder_grad_norm ** 0.5
        stats['embedder_grad_norm'] = embedder_grad_norm
        print(f"  Embedder Gradient norm: {embedder_grad_norm:.4f}")
    
    return stats

def log_training_stats(epoch, loss, acc, stats=None, log_file=None):
    """
    Registra estadísticas de entrenamiento en un archivo y en la consola.
    
    Args:
        epoch (int): Época actual
        loss (float): Valor de pérdida
        acc (float): Precisión
        stats (dict, optional): Estadísticas adicionales para registrar
        log_file (str, optional): Ruta al archivo de log
    """
    message = f"[Episode {epoch}] Loss: {loss:.4f}, Acc: {acc*100:.2f}%"
    
    if stats:
        for key, value in stats.items():
            if isinstance(value, float):
                message += f", {key}: {value:.4f}"
            else:
                message += f", {key}: {value}"
    
    # Usar tqdm.write para no interferir con las barras de progreso
    tqdm.write(message)
    
    if log_file:
        with open(log_file, 'a') as f:
            f.write(message + '\n')

def create_experiment_tracker(save_dir, experiment_name):
    """
    Crea un seguidor de experimentos para mantener métricas y resultados.
    
    Args:
        save_dir (str): Directorio para guardar resultados
        experiment_name (str): Nombre del experimento
        
    Returns:
        dict: Objeto trackeador con métodos para registrar métricas
    """
    # Crear directorios
    os.makedirs(save_dir, exist_ok=True)
    for subdir in ["figures", "models", "logs", "task_examples"]:
        os.makedirs(os.path.join(save_dir, subdir), exist_ok=True)
    
    # Archivos de log
    train_log = os.path.join(save_dir, "logs", f"{experiment_name}_train.log")
    val_log = os.path.join(save_dir, "logs", f"{experiment_name}_validation.log")
    
    # Limpiar archivos de log existentes
    with open(train_log, 'w') as f:
        f.write(f"# Training log for experiment: {experiment_name}\n")
    with open(val_log, 'w') as f:
        f.write(f"# Validation log for experiment: {experiment_name}\n")
    
    # Inicializar métricas
    metrics = {
        "train_losses": [],
        "train_accs": [],
        "val_accs": [],
        "val_epochs": [],
        "kernel_params": [],
        "gradient_norms_hypernet": [],
        "gradient_norms_embedder": [],
        "start_time": time.time()
    }
    
    # Métodos del trackeador
    def log_train(epoch, loss, acc, extra_stats=None):
        metrics["train_losses"].append(loss)
        metrics["train_accs"].append(acc)
        log_training_stats(epoch, loss, acc, extra_stats, train_log)
    
    def log_validation(epoch, val_acc, val_std=None):
        metrics["val_accs"].append(val_acc)
        metrics["val_epochs"].append(epoch)
        
        message = f"\nValidation at epoch {epoch}: Acc = {val_acc*100:.2f}%"
        if val_std is not None:
            message += f" ± {val_std*100:.2f}%"
        
        tqdm.write(message)
        with open(val_log, 'a') as f:
            f.write(message + '\n')
    
    def log_kernel_params(stats):
        metrics["kernel_params"].append(stats)
    
    def log_gradient_norm(hypernet_norm, embedder_norm=None):
        metrics["gradient_norms_hypernet"].append(hypernet_norm)
        if embedder_norm is not None:
            metrics["gradient_norms_embedder"].append(embedder_norm)
    
    def save_metrics():
        metrics_file = os.path.join(save_dir, f"{experiment_name}_metrics.npz")
        
        # Añadir tiempo total de entrenamiento
        metrics["training_time"] = time.time() - metrics["start_time"]
        
        np.savez(
            metrics_file,
            train_losses=np.array(metrics["train_losses"]),
            train_accs=np.array(metrics["train_accs"]),
            val_accs=np.array(metrics["val_accs"]),
            val_epochs=np.array(metrics["val_epochs"]),
            kernel_params=metrics["kernel_params"],
            gradient_norms_hypernet=metrics["gradient_norms_hypernet"],
            gradient_norms_embedder=metrics["gradient_norms_embedder"] if metrics["gradient_norms_embedder"] else None,
            training_time=metrics["training_time"]
        )
        return metrics_file
    
    # Definir el objeto trackeador
    tracker = {
        "log_train": log_train,
        "log_validation": log_validation,
        "log_kernel_params": log_kernel_params,
        "log_gradient_norm": log_gradient_norm,
        "save_metrics": save_metrics,
        "train_log": train_log,
        "val_log": val_log,
        "save_dir": save_dir,
        "experiment_name": experiment_name
    }
    
    return tracker