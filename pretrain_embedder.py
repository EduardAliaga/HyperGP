import os
import sys
import random
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.embedder import ConvEmbedder, Classifier
from utils.experiment import ExperimentManager
from data.dataset import MiniImageNetDataset

def parse_args():
    """Parse command line arguments, the default parameters are the ones used in the results of the report."""
    parser = argparse.ArgumentParser(description='Pre-entrenar el embedder Conv4')
    parser.add_argument('--dataset', type=str, default='miniimagenet',
                        choices=['miniimagenet', 'cub'],
                        help='Dataset to use (default: miniimagenet)')
    parser.add_argument('--dataset-path', type=str, required=True,
                        help='Path to the dataset')
    parser.add_argument('--epochs', type=int, default=90,
                        help='Number of epochs to train (default: 400)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for training (default: 128)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--feature-dim', type=int, default=64,
                        help='Feature dimension (default: 64)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight decay (default: 5e-4)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--checkpoint-interval', type=int, default=50,
                        help='Save checkpoint every N epochs (default: 50)')
    parser.add_argument('--no-data-aug', action='store_true',
                        help='Disable data augmentation')
    parser.add_argument('--save-dir', type=str, default='pretrained_embedders',
                        help='Directory to save models (default: pretrained_embedders)')
    
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def validate(embedder, classifier, val_loader, criterion, device):
    embedder.eval()
    classifier.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device).long()
            z = embedder(x)
            logits = classifier(z)
            
            loss = criterion(logits, y)
            val_loss += loss.item() * x.size(0)
            
            _, predicted = logits.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
    
    val_loss /= total
    val_acc = correct / total
    
    return val_loss, val_acc

def plot_training_history(train_losses, train_accs, val_losses, val_accs, save_path):
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train')
    plt.plot(val_accs, label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    args = parse_args()
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = vars(args)
    exp_name = f"pretrain_{args.dataset}_dim{args.feature_dim}_lr{args.lr}"
    experiment = ExperimentManager(config, exp_name)

    if args.no_data_aug:
        train_transform = transforms.Compose([
            transforms.Resize((84, 84)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((84, 84)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    test_transform = transforms.Compose([
        transforms.Resize((84, 84)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    if args.dataset == 'miniimagenet':
        full_dataset = MiniImageNetDataset(args.dataset_path, transform=train_transform)
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented yet")

    num_samples = len(full_dataset)
    num_train = int(0.9 * num_samples)
    num_val = num_samples - num_train
    
    train_dataset, val_dataset = random_split(
        full_dataset, [num_train, num_val],
        generator=torch.Generator().manual_seed(args.seed)
    )

    val_dataset.dataset = MiniImageNetDataset(
        args.dataset_path, 
        synsets=full_dataset.synsets,
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Dataset: {args.dataset}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Number of classes: {len(full_dataset.classes)}")

    embedder = ConvEmbedder(args.feature_dim, 4).to(device)  # 4 capas convolucionales
    classifier = Classifier(args.feature_dim, len(full_dataset.classes)).to(device)

    optimizer = optim.Adam(
        list(embedder.parameters()) + list(classifier.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5,
        patience=5, verbose=True
    )
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_val_acc = 0.0
    
    print(f"Starting training for {args.epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        embedder.train()
        classifier.train()
        train_loss = 0.0
        correct = 0
        total = 0
 
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for x, y in progress_bar:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            z = embedder(x)
            logits = classifier(z)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * x.size(0)
            _, predicted = logits.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': 100 * correct / total
            })
        
        train_loss = train_loss / total
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        val_loss, val_acc = validate(embedder, classifier, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        scheduler.step(val_acc)
        
        experiment.record_metrics({
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        log_msg = f"Epoch {epoch+1}/{args.epochs}: "
        log_msg += f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%, "
        log_msg += f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%"
        experiment.log_message(log_msg)
        
        if (epoch + 1) % args.checkpoint_interval == 0 or epoch == 0:
            checkpoint_path = experiment.get_checkpoint_path(epoch + 1)
            torch.save({
                'n_layers': 4,
                'feature_dim': args.feature_dim,
                'embedder_state_dict': embedder.state_dict(),
                'classifier_state_dict': classifier.state_dict(),
                'epoch': epoch + 1,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'optimizer_state_dict': optimizer.state_dict()
            }, checkpoint_path)
            experiment.log_message(f"Checkpoint saved to {checkpoint_path}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(
                args.save_dir, f"best_{args.dataset}_dim{args.feature_dim}.pth"
            )
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save({
                'n_layers': 4,
                'feature_dim': args.feature_dim,
                'embedder_state_dict': embedder.state_dict(),
                'epoch': epoch + 1,
                'val_acc': val_acc
            }, best_model_path)
            experiment.log_message(f"New best model saved to {best_model_path}")
    
    final_model_path = os.path.join(
        args.save_dir, f"final_{args.dataset}_dim{args.feature_dim}.pth"
    )
    torch.save({
        'n_layers': 4,
        'feature_dim': args.feature_dim,
        'embedder_state_dict': embedder.state_dict(),
        'epoch': args.epochs,
        'val_acc': val_accs[-1]
    }, final_model_path)
    
    history_plot_path = os.path.join(experiment.get_figures_dir(), 'training_history.png')
    plot_training_history(train_losses, train_accs, val_losses, val_accs, history_plot_path)
    
    experiment.record_metrics({
        'best_val_acc': best_val_acc,
        'final_val_acc': val_accs[-1],
        'training_time': time.time() - start_time
    })
    experiment.save_metrics()
    
    print(f"\nTraining completed in {(time.time() - start_time)/60:.2f} minutes")
    print(f"Best validation accuracy: {best_val_acc*100:.2f}%")
    print(f"Final models saved to {args.save_dir}")
    print(f"Experiment logs and metrics saved to {experiment.exp_dir}")

if __name__ == "__main__":
    main()