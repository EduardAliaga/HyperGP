import os
import sys
import argparse
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.embedder import ConvEmbedder
from models.hypernet import HyperNet
from utils.kernel import rbf_kernel, solve_gp_system
from utils.visualization import save_task_example, plot_accuracy_distribution, plot_kernel_params, plot_calibration,visualize_kernel_weights
from utils.metrics import compute_confidence_metrics, expected_calibration_error, compute_per_class_metrics
from data.dataset import load_cub_images, build_class_dict, split_classes
from data.sampling import sample_meta_task

def parse_args():
    """Parse command line arguments, the default parameters are the ones used in the results of the report."""
    parser = argparse.ArgumentParser(description='Evaluate HyperGP model')
    # Model parameters
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--n-way', type=int, default=None,
                        help='Number of classes per task (default: from model)')
    parser.add_argument('--k-shot', type=int, default=None,
                        help='Number of support examples per class (default: from model)')
    parser.add_argument('--q-query', type=int, default=16,
                        help='Number of query examples per class (default: 16)')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='cub',
                        choices=['cub', 'miniimagenet'],
                        help='Dataset to use (default: cub)')
    parser.add_argument('--dataset-path', type=str, required=True,
                        help='Path to the dataset')
    
    # Evaluation parameters
    parser.add_argument('--num-episodes', type=int, default=600,
                        help='Number of test episodes (default: 600)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--save-dir', type=str, default='eval_results',
                        help='Directory to save results (default: eval_results)')
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='Experiment name (default: auto-generated)')
    
    # Visualization options
    parser.add_argument('--num-examples', type=int, default=10,
                        help='Number of example tasks to visualize (default: 10)')
    
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def save_task_examples(sx, sy, qx, qy, preds, acc, task_idx, save_dir):
    """
    Saves visualizations of example tasks.

    Args:
        sx, sy, qx, qy: Task data
        preds: Model predictions
        acc: Accuracy on this task
        task_idx: Index of the task
        save_dir: Directory to save to
        
    Returns:
        str: Path of the saved file
    """
    output_path = os.path.join(
        save_dir, 
        f"task_{task_idx}_acc_{acc:.2f}.png"
    )
    return save_task_example(sx, sy, qx, qy, preds, output_path)

def format_results_table(results):
    table = "\n" + "="*60 + "\n"
    table += f"EVALUATION RESULTS\n"
    table += "="*60 + "\n"
    
    table += f"Model: HyperGP {results['n_way']}-way {results['k_shot']}-shot\n"
    table += f"Embedder: {'Fine-tuned' if results['finetune_embedder'] else 'Frozen'}\n"
    table += f"Number of test episodes: {results['num_episodes']}\n"
    table += f"Evaluation time: {results['time_total']:.2f} seconds " 
    table += f"({results['time_per_episode']*1000:.2f} ms per episode)\n"
    
    table += "\nAccuracy:\n"
    table += f"  Mean:   {results['accuracy']*100:.2f}%\n"
    table += f"  95% CI: ±{results['confidence_interval']*100:.2f}%\n"
    table += f"  Std:    {results['std']*100:.2f}%\n"
    table += f"  Median: {results['median']*100:.2f}%\n"
    table += f"  Min:    {results['min']*100:.2f}%\n"
    table += f"  Max:    {results['max']*100:.2f}%\n"
    
    if 'ece' in results:
        table += f"\nCalibration:\n"
        table += f"  ECE: {results['ece']:.4f}\n"

    table += f"\nPerformance:\n"
    table += f"  Avg. inference time: {results['time_per_episode']*1000:.2f} ms\n"
    
    table += "="*60 + "\n"
    return table

def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.experiment_name is None:
        args.experiment_name = f"eval_{args.dataset}_{os.path.basename(args.model_path).split('.')[0]}"
    
    save_dir = os.path.join(args.save_dir, args.experiment_name)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "task_examples"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "figures"), exist_ok=True)

    print(f"Loading model from: {args.model_path}")
    try:
        model_state = torch.load(args.model_path, map_location=device)
    except Exception as e:
        try:
            print(f"Failed to load with default options: {e}")
            model_state = torch.load(args.model_path, map_location=device, weights_only=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    if args.n_way is None:
        args.n_way = model_state.get('n_way', 5)
        print(f"Using N-way from model: {args.n_way}")
    
    if args.k_shot is None:
        args.k_shot = model_state.get('k_shot', 5)
        print(f"Using K-shot from model: {args.k_shot}")
    
    feature_dim = model_state.get('feature_dim', 64)
    hypernet_layers = model_state.get('hypernet_layers', 3)
    hypernet_hidden = model_state.get('hypernet_hidden', 512)
    finetune_embedder = model_state.get('finetune_embedder', False)

    embedder = ConvEmbedder(feature_dim, 4).to(device)
    hypernet = HyperNet(
        NK=args.n_way * args.k_shot, 
        D=feature_dim, 
        H=hypernet_hidden,
        n_hidden=hypernet_layers,
        N_way=args.n_way
    ).to(device)
    
    embedder.load_state_dict(model_state['embedder_state_dict'])
    hypernet.load_state_dict(model_state['hypernet_state_dict'])

    embedder.eval()
    hypernet.eval()

    if args.dataset == 'cub':
        print("Loading CUB dataset...")

        images, labels = load_cub_images(args.dataset_path)
        
        transform = torch.transforms.Compose([
            torch.transforms.Resize((84, 84)),
            torch.transforms.ToTensor(),
            torch.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        train_cls, test_cls = split_classes(labels, ratio=0.8, seed=args.seed)

        full_dict = build_class_dict(images, labels, transform, args.dataset_path)
        meta_test_dict = {c: full_dict[c] for c in test_cls}
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented yet")
    
    print(f"Evaluation classes: {len(test_cls)}")
    
    print(f"Starting evaluation on {args.num_episodes} episodes...")

    test_accs = []
    task_confidences = []
    test_times = []
    task_stats = []
    kernel_param_stats = []
    
    with torch.no_grad():
        for task_idx in tqdm(range(args.num_episodes)):
            task_start = time.time()

            (sx, sy), (qx, qy) = sample_meta_task(meta_test_dict, args.n_way, args.k_shot, args.q_query, device)

            z_s = embedder(sx)
            z_q = embedder(qx)
            Y_s = F.one_hot(sy, args.n_way).float()

            ell, sf, sn = hypernet(z_s, Y_s)
            
            K_ss = rbf_kernel(z_s, z_s, ell, sf) + (sn**2) * torch.eye(args.n_way * args.k_shot, device=device)
            K_sq = rbf_kernel(z_s, z_q, ell, sf)

            mu_q = solve_gp_system(K_ss, Y_s, K_sq)
            probs = F.softmax(mu_q, dim=1)
            preds = mu_q.argmax(1).cpu()
            confidences = probs.max(1)[0].cpu()

            accuracy = (preds == qy.cpu()).float().mean().item()
            test_accs.append(accuracy)
            
            task_time = time.time() - task_start
            test_times.append(task_time)
            
            task_confidences.append(confidences.numpy())
            
            kernel_param_stats.append({
                'ell_min': ell.min().item(),
                'ell_mean': ell.mean().item(),
                'ell_max': ell.max().item(),
                'sf': sf.item(),
                'sn': sn.item(),
            })
            
            task_stats.append({
                'task_id': task_idx,
                'accuracy': accuracy,
                'time': task_time
            })

            if task_idx < args.num_examples:
                example_path = save_task_examples(
                    sx, sy, qx, qy, preds, accuracy, task_idx, 
                    os.path.join(save_dir, "task_examples")
                )

    mean_acc = np.mean(test_accs)
    std_acc = np.std(test_accs)
    ci95 = 1.96 * std_acc / np.sqrt(len(test_accs))
    median_acc = np.median(test_accs)
    min_acc = np.min(test_accs)
    max_acc = np.max(test_accs)
    
    mean_time = np.mean(test_times)
    total_time = np.sum(test_times)

    all_confidences = np.concatenate([conf for conf in task_confidences])
    all_correctness = np.concatenate([
        (preds == qy.cpu()).float().numpy() 
        for preds, (_, _), (_, qy) in zip(
            [a.argmax(1).cpu() for a in [mu_q] * args.num_episodes],
            [(None, None)] * args.num_episodes,
            [(None, qy) for (_, _), (_, qy) in [
                sample_meta_task(
                    meta_test_dict, args.n_way, args.k_shot, args.q_query, device
                ) 
                for _ in range(args.num_episodes)
            ]]
        )
    ])
    

    ece, bins_data = expected_calibration_error(all_confidences, all_correctness)

    results = {
        'n_way': args.n_way,
        'k_shot': args.k_shot,
        'finetune_embedder': finetune_embedder,
        'num_episodes': args.num_episodes,
        'accuracy': mean_acc,
        'std': std_acc,
        'confidence_interval': ci95,
        'median': median_acc,
        'min': min_acc,
        'max': max_acc,
        'time_per_episode': mean_time,
        'time_total': total_time,
        'ece': ece,
        'feature_dim': feature_dim,
        'hypernet_layers': hypernet_layers,
        'hypernet_hidden': hypernet_hidden
    }

    figures_dir = os.path.join(save_dir, "figures")
    acc_dist_path = os.path.join(figures_dir, "accuracy_distribution.png")
    plot_accuracy_distribution(test_accs, acc_dist_path)

    calib_path = os.path.join(figures_dir, "calibration.png")
    plot_calibration(all_confidences, all_correctness, calib_path)

    ell_mins = [stats['ell_min'] for stats in kernel_param_stats]
    ell_means = [stats['ell_mean'] for stats in kernel_param_stats]
    ell_maxs = [stats['ell_max'] for stats in kernel_param_stats]
    sfs = [stats['sf'] for stats in kernel_param_stats]
    sns = [stats['sn'] for stats in kernel_param_stats]
    
    np.savez(
        os.path.join(save_dir, "kernel_param_stats.npz"),
        ell_mins=ell_mins,
        ell_means=ell_means,
        ell_maxs=ell_maxs,
        sfs=sfs,
        sns=sns
    )
    
    avg_ell = np.mean([stats['ell_mean'] for stats in kernel_param_stats])
    if avg_ell > 0:
        avg_ell_tensor = torch.tensor(ell_means, device=device).mean(0)
        vis_path = os.path.join(figures_dir, "feature_importance.png")
        visualize_kernel_weights(avg_ell_tensor, output_path=vis_path)
    
    np.save(os.path.join(save_dir, "test_accs.npy"), test_accs)
    np.save(os.path.join(save_dir, "task_times.npy"), test_times)
    
    table = format_results_table(results)
    print(table)
    
    with open(os.path.join(save_dir, "results_summary.txt"), 'w') as f:
        f.write(table)

    import json
    with open(os.path.join(save_dir, "results.json"), 'w') as f:
        results_json = {k: float(v) if isinstance(v, np.floating) else v 
                        for k, v in results.items()}
        json.dump(results_json, f, indent=2)
    
    print(f"Evaluation completed. Results saved to: {save_dir}")

if __name__ == "__main__":
    main()