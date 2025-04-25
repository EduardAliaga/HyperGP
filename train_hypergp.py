import os
import sys
import argparse
import random
import time
import numpy as np
import torch
from torch.optim.lr_scheduler import MultiStepLR
import torch.optim as optim
from tqdm import tqdm
from torchvision import transforms


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.embedder import ConvEmbedder
from models.hypernet import HyperNet
from utils.kernel import rbf_kernel, solve_gp_system
from utils.monitoring import monitor_params, create_experiment_tracker
from utils.visualization import save_task_example, plot_learning_curves, plot_kernel_params, plot_accuracy_distribution,plot_calibration
from data.dataset import load_cub_images, build_class_dict, split_classes
from data.sampling import sample_meta_task
import torch.nn.functional as F

def parse_args():
    """Parse command line arguments, the default parameters are the ones used in the results of the report."""
    parser = argparse.ArgumentParser(description='Train HyperGP model for few-shot learning')
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='cub',
                        choices=['cub', 'miniimagenet'],
                        help='Dataset to use (default: cub)')
    parser.add_argument('--dataset-path', type=str, required=True,
                        help='Path to the dataset')
    
    # Model parameters
    parser.add_argument('--n-way', type=int, default=5,
                        help='Number of classes per task (default: 5)')
    parser.add_argument('--k-shot', type=int, default=5,
                        help='Number of support examples per class (default: 5)')
    parser.add_argument('--q-query', type=int, default=16,
                        help='Number of query examples per class (default: 16)')
    parser.add_argument('--feature-dim', type=int, default=64,
                        help='Feature dimension (default: 64)')
    parser.add_argument('--hypernet-layers', type=int, default=3,
                        help='Number of layers in hypernetwork (default: 3)')
    parser.add_argument('--hypernet-hidden', type=int, default=512,
                        help='Hidden dimension in hypernetwork (default: 512)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=4000,
                        help='Number of meta-training episodes (default: 4000)')
    parser.add_argument('--lr-hypernet', type=float, default=1e-3,
                        help='Learning rate for hypernetwork (default: 1e-3)')
    parser.add_argument('--lr-embedder', type=float, default=1e-4,
                        help='Learning rate for embedder when fine-tuning (default: 1e-4)')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay (default: 1e-5)')
    parser.add_argument('--max-grad-norm', type=float, default=1.0,
                        help='Max gradient norm for clipping (default: 1.0)')
    parser.add_argument('--lr-milestones', type=int, nargs='+', default=[1000, 2000, 3000],
                        help='Epochs at which to reduce the learning rate (default: 1000 2000 3000)')
    parser.add_argument('--lr-gamma', type=float, default=0.3,
                        help='Learning rate decay factor (default: 0.3)')
    parser.add_argument('--finetune-embedder', action='store_true',
                        help='Whether to fine-tune the embedder')
    
    # Embedder parameters
    parser.add_argument('--pretrained-embedder', type=str, required=True,
                        help='Path to pretrained embedder model')
    
    # Validation and logging
    parser.add_argument('--val-interval', type=int, default=200,
                        help='Validate every N epochs (default: 200)')
    parser.add_argument('--val-episodes', type=int, default=50,
                        help='Number of episodes for validation (default: 50)')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='Log training stats every N epochs (default: 100)')
    
    # Experiment parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--save-dir', type=str, default='experiments',
                        help='Directory to save results (default: experiments)')
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='Experiment name (default: auto-generated)')
    
    return parser.parse_args()

def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    args = parse_args()
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = vars(args)
    if args.experiment_name is None:
        args.experiment_name = (
            f"hypergp_{args.dataset}_{args.n_way}way_{args.k_shot}shot"
            f"_finetune_{args.finetune_embedder}"
        )
    tracker = create_experiment_tracker(args.save_dir, args.experiment_name)
    
    if args.dataset == 'cub':
        print("Loading CUB dataset...")
        images_file = os.path.join(args.dataset_path, "images.txt")
        labels_file = os.path.join(args.dataset_path, "image_class_labels.txt")
        image_dir = os.path.join(args.dataset_path, "images")
        
        # transform = torch.transforms.Compose([
        #     torch.transforms.Resize((84, 84)),
        #     torch.transforms.ToTensor(),
        #     torch.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        # ])
        transform = transforms.Compose([
            transforms.Resize((84, 84)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
  
        images, labels = load_cub_images(args.dataset_path)
        train_cls, test_cls = split_classes(labels, ratio=0.8, seed=args.seed)
        
        full_dict = build_class_dict(images, labels, transform, args.dataset_path)
        meta_train_dict = {c: full_dict[c] for c in train_cls}
        meta_test_dict = {c: full_dict[c] for c in test_cls}
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented yet")
    
    print(f"Training classes: {len(train_cls)}, Test classes: {len(test_cls)}")
    
    print(f"Loading pretrained embedder from: {args.pretrained_embedder}")
    embedder_ckpt = torch.load(args.pretrained_embedder, map_location=device)
    embedder = ConvEmbedder(
        embedder_ckpt['feature_dim'], 
        embedder_ckpt.get('n_layers', 4)
    ).to(device)
    embedder.load_state_dict(embedder_ckpt['embedder_state_dict'])
    
    if args.finetune_embedder:
        embedder.train()
        print("Embedder set for fine-tuning")
    else:
        embedder.eval()
        for param in embedder.parameters():
            param.requires_grad = False
        print("Embedder frozen (no fine-tuning)")
    
    print("Initializing HyperNet")
    hypernet = HyperNet(
        NK=args.n_way * args.k_shot,
        D=args.feature_dim,
        H=args.hypernet_hidden,
        n_hidden=args.hypernet_layers,
        N_way=args.n_way
    ).to(device)
    
    if args.finetune_embedder:
        param_groups = [
            {'params': hypernet.parameters(), 'lr': args.lr_hypernet},
            {'params': embedder.parameters(), 'lr': args.lr_embedder}
        ]
        print(f"Optimizing both - HyperNet (LR: {args.lr_hypernet}), Embedder (LR: {args.lr_embedder})")
    else:
        param_groups = [
            {'params': hypernet.parameters(), 'lr': args.lr_hypernet}
        ]
        print(f"Optimizing only hypernetwork (LR: {args.lr_hypernet})")
    
    optimizer = optim.Adam(param_groups, weight_decay=args.weight_decay)
    
    scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=args.lr_gamma)
    
    best_val_acc = 0.0
    train_losses = []
    train_accs = []
    val_epochs = []
    val_accs = []
    kernel_params = []
    gradient_norms_hypernet = []
    gradient_norms_embedder = []
    
    print(f"Starting meta-training ({args.epochs} episodes)")
    start_time = time.time()
    
    for ep in tqdm(range(1, args.epochs + 1)):
        hypernet.train()
        if args.finetune_embedder:
            embedder.train()
        (sx, sy), (qx, qy) = sample_meta_task(meta_train_dict, args.n_way, args.k_shot, args.q_query, device)

        if args.finetune_embedder:
            z_s = embedder(sx)
            z_q = embedder(qx)
        else:
            with torch.no_grad():
                z_s = embedder(sx)
                z_q = embedder(qx)
        Y_s = F.one_hot(sy, args.n_way).float()
        
        ell, sf, sn = hypernet(z_s, Y_s)
        K_ss = rbf_kernel(z_s, z_s, ell, sf) + (sn**2) * torch.eye(args.n_way * args.k_shot, device=device)
        K_sq = rbf_kernel(z_s, z_q, ell, sf)
        mu_q = solve_gp_system(K_ss, Y_s, K_sq)

        loss = F.cross_entropy(mu_q, qy)
        acc = (mu_q.argmax(1) == qy).float().mean().item()

        train_losses.append(loss.item())
        train_accs.append(acc)
        
        optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(hypernet.parameters(), args.max_grad_norm)
        if args.finetune_embedder:
            torch.nn.utils.clip_grad_norm_(embedder.parameters(), args.max_grad_norm)
        
        optimizer.step()
        scheduler.step()
        
        if ep % args.log_interval == 0:
            param_stats = monitor_params(ep, ell, sf, sn, hypernet, embedder, args.finetune_embedder)
            kernel_params.append(param_stats)
            
            hypernet_grad_norm = param_stats['hypernet_grad_norm']
            gradient_norms_hypernet.append({
                'epoch': ep, 
                'norm': hypernet_grad_norm
            })
            
            if args.finetune_embedder:
                embedder_grad_norm = param_stats['embedder_grad_norm']
                gradient_norms_embedder.append({
                    'epoch': ep, 
                    'norm': embedder_grad_norm
                })

            log_msg = f"Episode {ep}/{args.epochs}: Loss={loss.item():.4f}, Acc={acc*100:.2f}%"
            tracker["log_train"](ep, loss.item(), acc)
        
        if ep % args.val_interval == 0:
            hypernet.eval()
            embedder.eval()
            val_accs_this_epoch = []
            
            print(f"\nValidating at episode {ep}")
            with torch.no_grad():
                for task_idx in range(args.val_episodes):
                    (sx, sy), (qx, qy) = sample_meta_task(
                        meta_test_dict, args.n_way, args.k_shot, args.q_query, device
                    )
                    z_s = embedder(sx)
                    z_q = embedder(qx)
                    Y_s = F.one_hot(sy, args.n_way).float()
                    
                    ell, sf, sn = hypernet(z_s, Y_s)
                    K_ss = rbf_kernel(z_s, z_s, ell, sf) + (sn**2) * torch.eye(
                        args.n_way * args.k_shot, device=device
                    )
                    K_sq = rbf_kernel(z_s, z_q, ell, sf)
                    
                    mu_q = solve_gp_system(K_ss, Y_s, K_sq)
                    preds = mu_q.argmax(1).cpu()
                    task_acc = (preds == qy.cpu()).float().mean().item()
                    val_accs_this_epoch.append(task_acc)
                    if task_idx == 0:
                        example_path = os.path.join(
                            tracker["save_dir"], 
                            "task_examples", 
                            f"task_example_epoch_{ep}.png"
                        )
                        save_task_example(sx, sy, qx, qy, preds, example_path)
            
            val_acc_mean = np.mean(val_accs_this_epoch)
            val_acc_std = np.std(val_accs_this_epoch)
        
            val_epochs.append(ep)
            val_accs.append(val_acc_mean)
            
            print(f"Validation Acc = {val_acc_mean*100:.2f}% ± {val_acc_std*100:.2f}%")
            tracker["log_validation"](ep, val_acc_mean, val_acc_std)
            
            is_best = val_acc_mean > best_val_acc
            if is_best:
                best_val_acc = val_acc_mean
                best_model_path = os.path.join(
                    tracker["save_dir"],
                    "models",
                    f"best_{args.experiment_name}.pth"
                )
                torch.save({
                    'embedder_state_dict': embedder.state_dict(),
                    'hypernet_state_dict': hypernet.state_dict(),
                    'epoch': ep,
                    'val_acc': val_acc_mean,
                    'val_acc_std': val_acc_std,
                    'n_way': args.n_way,
                    'k_shot': args.k_shot,
                    'finetune_embedder': args.finetune_embedder,
                    'lr_hypernet': args.lr_hypernet,
                    'lr_embedder': args.lr_embedder if args.finetune_embedder else None,
                    'feature_dim': args.feature_dim,
                    'hypernet_layers': args.hypernet_layers,
                    'hypernet_hidden': args.hypernet_hidden
                }, best_model_path)
                print(f"New best model saved! (Acc: {val_acc_mean*100:.2f}%)")
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    np.savez(
        os.path.join(tracker["save_dir"], f"{args.experiment_name}_metrics.npz"),
        train_losses=np.array(train_losses),
        train_accs=np.array(train_accs),
        val_accs=np.array(val_accs),
        val_epochs=np.array(val_epochs),
        kernel_params=kernel_params,
        gradient_norms_hypernet=gradient_norms_hypernet,
        gradient_norms_embedder=gradient_norms_embedder if gradient_norms_embedder else None,
        training_time=training_time
    )


    figures_dir = os.path.join(tracker["save_dir"], "figures")
    
    plot_learning_curves(train_losses, val_epochs, val_accs,os.path.join(figures_dir, f"{args.experiment_name}_learning_curves.png"))
    plot_kernel_params(kernel_params, gradient_norms_hypernet, gradient_norms_embedder if args.finetune_embedder else None,os.path.join(figures_dir, f"{args.experiment_name}_kernel_params.png"))
    
    print(f"Best validation accuracy: {best_val_acc*100}%")
    print(f"All results saved to: {os.path.join(tracker['save_dir'])}")

if __name__ == "__main__":
    main()