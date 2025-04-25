import os
import numpy as np
import torch
import time
from tqdm import tqdm
import json

class ExperimentManager:
    """
    Class to manage experiments, saving results,
    checkpoints, and statistics in an organized manner.
    """
    def __init__(self, config, experiment_name=None):
        """
        Initializes the ExperimentManager with a given configuration.
        
        Args:
            config (dict): Experiment configuration
            experiment_name (str, optional): Experiment name
        """
        self.config = config
        
        if experiment_name is None:
            self.experiment_name = f"exp_{int(time.time())}"
        else:
            self.experiment_name = experiment_name
        
        self.base_dir = config.get('save_dir', 'experiments')
        self.exp_dir = os.path.join(self.base_dir, self.experiment_name)
        
        os.makedirs(self.exp_dir, exist_ok=True)
        for subdir in ['models', 'figures', 'logs', 'checkpoints', 'results']:
            os.makedirs(os.path.join(self.exp_dir, subdir), exist_ok=True)
        
        self.save_config()
        
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
        """Saves the experiment configuration to a JSON file."""

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
        Returns the path to save/load a model.
        
        Args:
            filename (str, optional)
            
        Returns:
            str: path to the model file
        """
        if filename is None:
            filename = f"model_epoch_{self.epoch}.pth"
        return os.path.join(self.exp_dir, 'models', filename)
    
    def get_checkpoint_path(self, epoch=None):
        """
        Returns the path to save/load a checkpoint.
        
        Args:
            epoch (int, optional): Checkpoint epoch
            
        Returns:
            str: path to the checkpoint file
        """
        if epoch is None:
            epoch = self.epoch
        return os.path.join(self.exp_dir, 'checkpoints', f"checkpoint_epoch_{epoch}.pth")
    
    def save_checkpoint(self, state, is_best=False):
        """
        Saves the current state of the model and optimizer.
        
        Args:
            state (dict): Saving state (model, optimizer, epoch, etc.)
            is_best (bool, optional): If this is the best model so far
            
        Returns:
            str: path to the saved checkpoint
        """
        checkpoint_path = self.get_checkpoint_path()
        torch.save(state, checkpoint_path)
        
        if is_best:
            best_model_path = self.get_model_path('best_model.pth')
            torch.save(state, best_model_path)
            
        return checkpoint_path
    
    def load_checkpoint(self, path=None, map_location=None):
        """
        Loads a checkpoint from the specified path.
        
        Args:
            path (str, optional): Path to the checkpoint file
            map_location: Argument for torch.load
            
        Returns:
            dict: Loaded state dictionary
        """
        if path is None:
            path = self.get_checkpoint_path()
        
        try:
            state = torch.load(path, map_location=map_location, weights_only=False)
        except:
            state = torch.load(path, map_location=map_location)
            
        if 'epoch' in state:
            self.epoch = state['epoch']
        if 'best_val_acc' in state:
            self.best_val_acc = state['best_val_acc']
            
        return state
    
    def record_metrics(self, metrics_dict):
        """
        Register metrics for the current experiment.
        
        Args:
            metrics_dict (dict): Metrics to register
        """
        for key, value in metrics_dict.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
    
    def save_metrics(self):
        """
        Saves the metrics to a file.
        
        Returns:
            str: path to the saved metrics file
        """
        self.metrics['training_time'] = time.time() - self.metrics['start_time']
        
        metrics_path = os.path.join(self.exp_dir, 'results', 'metrics.npz')
        np.savez(metrics_path, **self.metrics)
        
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
        Writes a message to the log file and prints it to the console.
        
        Args:
            message (str)
            log_type (str): (train, val, test)
        """
        log_path = os.path.join(self.exp_dir, 'logs', f"{log_type}.log")
        with open(log_path, 'a') as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
        
        tqdm.write(message)
    
    def update_epoch(self, increment=1):
        """
        Updates the current epoch.
        
        Args:
            increment (int, optional): Number of epochs to increment
        """
        self.epoch += increment
        
    def update_best_val_acc(self, val_acc):
        """
        Updates the best validation accuracy if the new accuracy is better.
        
        Args:
            val_acc (float): New validation accuracy
            
        Returns:
            bool: If the new accuracy is better than the previous best
        """
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            return True
        return False
    
    def get_figures_dir(self):
        """Returns the directory for saving figures."""
        return os.path.join(self.exp_dir, 'figures')
    
    def get_results_dir(self):
        """Returns the directory for saving results."""
        return os.path.join(self.exp_dir, 'results')
    
    def get_logs_dir(self):
        """Returns the directory for saving logs."""
        return os.path.join(self.exp_dir, 'logs')