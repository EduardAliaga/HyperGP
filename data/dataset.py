import os
import random
import torch
from PIL import Image
from torch.utils.data import Dataset
from collections import defaultdict

class CUBDataset:
    """
    Functions to load and process images from the CUB-200-2011 dataset.
    """
    def __init__(self, dataset_dir):
        """
        Args:
            dataset_dir (str): path to the dataset
        """
        self.dataset_dir = dataset_dir
        self.images_file = os.path.join(dataset_dir, "images.txt")
        self.labels_file = os.path.join(dataset_dir, "image_class_labels.txt")
        self.image_dir = os.path.join(dataset_dir, "images")
    
    def load_images(self):
        """
        Load images and labels from the dataset.
        
        Returns:
            tuple: names of the files and labels
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
        Build a dictionary of images by class.
        
        Args:
            imgs (list): List of file names
            lbs (list): List of labels
            transform: IMage transformations
            
        Returns:
            dict: {class: [transformed images]}
        """
        d = defaultdict(list)
        for fn, lb in zip(imgs, lbs):
            img = Image.open(os.path.join(self.image_dir, fn)).convert("RGB")
            d[lb].append(transform(img))
        return d


class MiniImageNetDataset(Dataset):
    """
    Functions to load and process images from the MiniImageNet.
    """
    def __init__(self, root, synsets=None, transform=None):
        """
        Args:
            root (str): path to the dataset
            synsets (list, optional): List of classes to use
            transform: Iameg transformations
        """
        self.root = root
        self.transform = transform
        self.samples = []
        self.targets = []
        self.classes = []
        
        if synsets is None:
            self.synsets = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        else:
            self.synsets = synsets
        
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.synsets)}
        
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
        """Returns the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Returns one sample from the dataset.
        
        Args:
            idx (int): sample index
            
        Returns:
            tuple: (transformed image, label)
        """
        img_path, target = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target


def load_cub_images(dataset_dir):
    """
    Wrapper to load the CUB images.
    
    Args:
        dataset_dir (str): path to the dataset
        
    Returns:
        tuple: List of file names and labels
    """
    cub_dataset = CUBDataset(dataset_dir)
    return cub_dataset.load_images()

def build_class_dict(imgs, lbs, transform, dataset_dir):
    """
    Build a dictionary of images by class.
    
    Args:
        imgs (list): List of file names
        lbs (list): List of labels
        transform: IMage transformations
        
    Returns:
        dict: {class: [transformed images]}
    """
    cub_dataset = CUBDataset(dataset_dir)
    return cub_dataset.build_class_dict(imgs, lbs, transform)

def split_classes(labels, ratio=0.8, seed=None):
    """
    Divides the classes into training and testing sets.
    
    Args:
        labels (list): List of labels
        ratio (float, optional): trainig and testing ratio
        seed (int, optional): Random seed for reproducibility
        
    Returns:
        tuple: training and testing classes
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