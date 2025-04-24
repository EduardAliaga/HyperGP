import os
import numpy as np
import torch
import time
from tqdm import tqdm
import json

class ExperimentManager:
    """
    Clase para gestionar experimentos, guardando resultados,
    checkpoints y estadísticas de manera organizada.
    """
    def __init__(self, config, experiment_name=None):
        """
        Inicializa el gestor de experimentos.
        
        Args:
            config (dict): Configuración del experimento
            experiment_name (str, optional): Nombre del experimento
        """
        self.config = config
        
        # Generar nombre de experimento si no se proporciona
        if experiment_name is None:
            self.experiment_name = f"exp_{int(time.time())}"
        else:
            self.experiment_name = experiment_name
        
        # Configurar directorios
        self.base_dir = config.get('save_dir', 'experiments')
        self.exp_dir = os.path.join(self.base_dir, self.experiment_name)
        
        # Crear subdirectorios
        os.makedirs(self.exp_dir, exist_ok=True)
        for subdir in ['models', 'figures', 'logs', 'checkpoints', 'results']:
            os.makedirs(os.path.join(self.exp_dir, subdir), exist_ok=True)
        
        # Guardar configuración
        self.save_config()
        
        # Inicializar contadores y registros
        self.epoch = 0
        self.best_val_acc = 0.0
        self.metrics = {
            'train_loss': [],
            'train_acc': [],
            'val_acc': [],
            'val_epochs': [],
            'start_time': time.time()
        }
    
    def save_config(self):
        """Guarda la configuración del experimento en formato JSON."""
        # Convertir valores no serializables a strings
        serializable_config = {}
        for k, v in self.config.items():
            if isinstance(v, torch.device):
                serializable_config[k] = str(v)
            else:
                serializable_config[k] = v
        
        config_path = os.path.join(self.exp_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(serializable_config, f, indent=2)
    
    def get_model_path(self, filename=None):
        """
        Retorna la ruta para guardar/cargar un modelo.
        
        Args:
            filename (str, optional): Nombre del archivo
            
        Returns:
            str: Ruta completa
        """
        if filename is None:
            filename = f"model_epoch_{self.epoch}.pth"
        return os.path.join(self.exp_dir, 'models', filename)
    
    def get_checkpoint_path(self, epoch=None):
        """
        Retorna la ruta para guardar/cargar un checkpoint.
        
        Args:
            epoch (int, optional): Época del checkpoint
            
        Returns:
            str: Ruta completa
        """
        if epoch is None:
            epoch = self.epoch
        return os.path.join(self.exp_dir, 'checkpoints', f"checkpoint_epoch_{epoch}.pth")
    
    def save_checkpoint(self, state, is_best=False):
        """
        Guarda un checkpoint del estado actual del entrenamiento.
        
        Args:
            state (dict): Estado a guardar
            is_best (bool, optional): Si es el mejor modelo hasta ahora
            
        Returns:
            str: Ruta del checkpoint guardado
        """
        # Guardar checkpoint normal
        checkpoint_path = self.get_checkpoint_path()
        torch.save(state, checkpoint_path)
        
        # Guardar también como mejor modelo si corresponde
        if is_best:
            best_model_path = self.get_model_path('best_model.pth')
            torch.save(state, best_model_path)
            
        return checkpoint_path
    
    def load_checkpoint(self, path=None, map_location=None):
        """
        Carga un checkpoint guardado.
        
        Args:
            path (str, optional): Ruta al checkpoint
            map_location: Argumento para torch.load
            
        Returns:
            dict: Estado cargado
        """
        if path is None:
            path = self.get_checkpoint_path()
        
        try:
            # Intentar cargar con seguridad para versiones recientes de PyTorch
            state = torch.load(path, map_location=map_location, weights_only=False)
        except:
            # Fallback a carga estándar
            state = torch.load(path, map_location=map_location)
            
        # Actualizar contadores internos
        if 'epoch' in state:
            self.epoch = state['epoch']
        if 'best_val_acc' in state:
            self.best_val_acc = state['best_val_acc']
            
        return state
    
    def record_metrics(self, metrics_dict):
        """
        Registra métricas de un paso de entrenamiento/validación.
        
        Args:
            metrics_dict (dict): Métricas a registrar
        """
        for key, value in metrics_dict.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
    
    def save_metrics(self):
        """
        Guarda todas las métricas registradas.
        
        Returns:
            str: Ruta del archivo de métricas
        """
        # Añadir tiempo total de entrenamiento
        self.metrics['training_time'] = time.time() - self.metrics['start_time']
        
        # Guardar métricas como archivo numpy
        metrics_path = os.path.join(self.exp_dir, 'results', 'metrics.npz')
        np.savez(metrics_path, **self.metrics)
        
        # También guardar algunos resultados clave en formato texto
        summary_path = os.path.join(self.exp_dir, 'results', 'summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Experiment: {self.experiment_name}\n")
            f.write(f"Training time: {self.metrics['training_time']:.2f} seconds\n")
            
            if 'val_acc' in self.metrics and len(self.metrics['val_acc']) > 0:
                best_acc_idx = np.argmax(self.metrics['val_acc'])
                best_acc = self.metrics['val_acc'][best_acc_idx]
                best_epoch = self.metrics['val_epochs'][best_acc_idx]
                f.write(f"Best validation accuracy: {best_acc*100:.2f}% at epoch {best_epoch}\n")
            
            if 'test_acc' in self.metrics:
                f.write(f"Test accuracy: {self.metrics['test_acc']*100:.2f}%")
                if 'test_acc_ci' in self.metrics:
                    f.write(f" ± {self.metrics['test_acc_ci']*100:.2f}%")
                f.write("\n")
        
        return metrics_path
    
    def log_message(self, message, log_type='train'):
        """
        Registra un mensaje en el archivo de log.
        
        Args:
            message (str): Mensaje a registrar
            log_type (str): Tipo de log (train, val, test)
        """
        log_path = os.path.join(self.exp_dir, 'logs', f"{log_type}.log")
        with open(log_path, 'a') as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
        
        # También mostrar en consola
        tqdm.write(message)
    
    def update_epoch(self, increment=1):
        """
        Incrementa el contador de épocas.
        
        Args:
            increment (int, optional): Cantidad a incrementar
        """
        self.epoch += increment
        
    def update_best_val_acc(self, val_acc):
        """
        Actualiza la mejor precisión de validación si es necesario.
        
        Args:
            val_acc (float): Nueva precisión de validación
            
        Returns:
            bool: Si se mejoró el mejor resultado
        """
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            return True
        return False
    
    def get_figures_dir(self):
        """Retorna el directorio para guardar figuras."""
        return os.path.join(self.exp_dir, 'figures')
    
    def get_results_dir(self):
        """Retorna el directorio para guardar resultados."""
        return os.path.join(self.exp_dir, 'results')
    
    def get_logs_dir(self):
        """Retorna el directorio para guardar logs."""
        return os.path.join(self.exp_dir, 'logs')