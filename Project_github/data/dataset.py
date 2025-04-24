import os
import random
import torch
from PIL import Image
from torch.utils.data import Dataset
from collections import defaultdict

class CUBDataset:
    """
    Clase para manejar el dataset CUB-200-2011.
    Proporciona funciones para cargar y procesar imágenes.
    """
    def __init__(self, dataset_dir):
        """
        Args:
            dataset_dir (str): Directorio que contiene el dataset CUB-200-2011
        """
        self.dataset_dir = dataset_dir
        self.images_file = os.path.join(dataset_dir, "images.txt")
        self.labels_file = os.path.join(dataset_dir, "image_class_labels.txt")
        self.image_dir = os.path.join(dataset_dir, "images")
    
    def load_images(self):
        """
        Carga los nombres de los archivos de imagen y sus etiquetas.
        
        Returns:
            tuple: Lista de nombres de archivo y lista de etiquetas
        """
        imgs, lbs = [], []
        with open(self.images_file) as f:
            for line in f:
                _, fn = line.strip().split()
                imgs.append(fn)
        with open(self.labels_file) as f:
            for line in f:
                _, lb = line.strip().split()
                lbs.append(int(lb))
        return imgs, lbs
    
    def build_class_dict(self, imgs, lbs, transform):
        """
        Construye un diccionario de imágenes por clase.
        
        Args:
            imgs (list): Lista de nombres de archivo
            lbs (list): Lista de etiquetas
            transform: Transformaciones para aplicar a las imágenes
            
        Returns:
            dict: Diccionario {clase: [imágenes transformadas]}
        """
        d = defaultdict(list)
        for fn, lb in zip(imgs, lbs):
            img = Image.open(os.path.join(self.image_dir, fn)).convert("RGB")
            d[lb].append(transform(img))
        return d


class MiniImageNetDataset(Dataset):
    """
    Dataset personalizado para MiniImageNet.
    """
    def __init__(self, root, synsets=None, transform=None):
        """
        Args:
            root (str): Directorio que contiene el dataset MiniImageNet
            synsets (list, optional): Lista de clases a incluir
            transform: Transformaciones para aplicar a las imágenes
        """
        self.root = root
        self.transform = transform
        self.samples = []
        self.targets = []
        self.classes = []
        
        if synsets is None:
            # Usar todas las clases disponibles
            self.synsets = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        else:
            self.synsets = synsets
        
        # Crear mapeo de etiquetas
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.synsets)}
        
        # Recopilar todas las muestras
        for class_name in self.synsets:
            class_dir = os.path.join(root, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            self.classes.append(class_name)
            class_idx = self.class_to_idx[class_name]
            
            for img_file in os.listdir(class_dir):
                if not img_file.lower().endswith(('png', 'jpg', 'jpeg')):
                    continue
                    
                img_path = os.path.join(class_dir, img_file)
                self.samples.append((img_path, class_idx))
                self.targets.append(class_idx)
    
    def __len__(self):
        """Retorna el número de muestras en el dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Retorna una muestra del dataset.
        
        Args:
            idx (int): Índice de la muestra
            
        Returns:
            tuple: (imagen transformada, etiqueta)
        """
        img_path, target = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target


# Funciones auxiliares
def load_cub_images(dataset_dir):
    """
    Wrapper para cargar imágenes CUB usando la clase CUBDataset.
    
    Args:
        dataset_dir (str): Directorio del dataset CUB
        
    Returns:
        tuple: Lista de nombres de archivo y lista de etiquetas
    """
    cub_dataset = CUBDataset(dataset_dir)
    return cub_dataset.load_images()

def build_class_dict(imgs, lbs, transform, dataset_dir):
    """
    Wrapper para construir un diccionario de imágenes por clase.
    
    Args:
        imgs (list): Lista de nombres de archivo
        lbs (list): Lista de etiquetas
        transform: Transformaciones para aplicar a las imágenes
        dataset_dir (str): Directorio del dataset
        
    Returns:
        dict: Diccionario {clase: [imágenes transformadas]}
    """
    cub_dataset = CUBDataset(dataset_dir)
    return cub_dataset.build_class_dict(imgs, lbs, transform)

def split_classes(labels, ratio=0.8, seed=None):
    """
    Divide las clases en conjuntos de entrenamiento y prueba.
    
    Args:
        labels (list): Lista de etiquetas de clase
        ratio (float, optional): Proporción para entrenamiento
        seed (int, optional): Semilla para reproducibilidad
        
    Returns:
        tuple: Clases de entrenamiento y clases de prueba
    """
    if seed is not None:
        random_state = random.getstate()
        random.seed(seed)
    
    uc = list(set(labels))
    random.shuffle(uc)
    n = int(len(uc)*ratio)
    result = (uc[:n], uc[n:])
    
    if seed is not None:
        random.setstate(random_state)
    
    return result