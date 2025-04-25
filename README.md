# HyperGP: Meta-Learning with Gaussian Processes and Hypernetworks

This repository implements **HyperGP**, a few-shot meta-learning model that combines a deep feature embedder with a Gaussian-process head whose kernel hyperparameters are predicted by a hypernetwork.

---

## Repository Structure

```
.
├── data/
│   ├── dataset.py   # CUB / mini-ImageNet loading & splitting
│   └── sampling.py  # Meta-task sampler
├── models/
│   ├── embedder.py  # ConvEmbedder backbone definition
│   └── hypernet.py  # HyperNet for GP hyperparameter prediction
├── utils/
│   ├── kernel.py    # RBF kernel & GP solver
│   ├── monitoring.py  # Logging, parameter tracking
│   └── visualization.py  # Plotting & example saving
├── train_hypergp.py  # Main training script
├── requirements.txt  # Python dependencies
└── README.md        # This file
```

---

## Installation

1. **Clone this repository**  
   ```bash
   git clone https://github.com/yourusername/hypergp.git
   cd hypergp
   ```

2. **Create and activate a virtual environment** (optional but recommended)
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   The main dependencies are:
   - numpy<2
   - torch
   - torchvision
   - tqdm
   - seaborn

## Preparing the Dataset

### CUB-200-2011

Download and extract the dataset so that you have a folder structure like:
```
data/CUB/CUB_200_2011/
├── images.txt
├── image_class_labels.txt
└── images/
    ├── 000001.jpg
    ├── 000002.jpg
    └── ...
```

When running the training script, point `--dataset-path` to `./CUB/CUB_200_2011`.

### mini-ImageNet (Optional)

If you wish to use mini-ImageNet, prepare the folder of splits accordingly and use `--dataset miniimagenet`.

## Usage

Run the main script with the required arguments:

```bash
python train_hypergp.py \
  --dataset cub \
  --dataset-path ./CUB/CUB_200_2011 \
  --pretrained-embedder path/to/embedder_checkpoint.pth \
  [--finetune-embedder] \
  [--n-way 5] [--k-shot 5] [--q-query 16] \
  [--epochs 4000] [--lr-hypernet 1e-3] [--lr-embedder 1e-4] \
  [--save-dir experiments] [--experiment-name my_run]
```

### Required Flags

- `--dataset-path`
  Path to your dataset root (e.g. data/CUB/CUB_200_2011).
- `--pretrained-embedder`
  Path to a pretrained ConvEmbedder checkpoint (e.g. pretrained_conv4d_mini.pth).

### Optional Flags

- `--finetune-embedder`
  Include this flag to fine-tune the embedder during meta-training.
- `--n-way`, `--k-shot`, `--q-query`
  Task configuration (defaults: 5, 5, 16).
- `--epochs`, `--lr-hypernet`, `--lr-embedder`, `--weight-decay`, etc.
  Training hyperparameters.
- `--save-dir`, `--experiment-name`
  Where to store checkpoints, logs, and figures.

Run `python train_hypergp.py -h` to see the full list of options.

### Outputs

After training, you will find under `experiments/<experiment-name>/`:
- `models/` – Saved checkpoints (`best_<name>.pth`).
- `figures/` – Learning curves, kernel parameter plots, accuracy distributions, calibration plots.
- `task_examples/` – Example support/query visualizations.
- `metrics.npz` – NumPy archive of train/validation losses, accuracies, kernel stats, gradient norms, and timing.

## Example

```bash
python train_hypergp.py \
  --dataset cub \
  --dataset-path data/CUB/CUB_200_2011 \
  --pretrained-embedder pretrained_conv4d_mini.pth \
  --finetune-embedder \
  --n-way 5 --k-shot 5 --q-query 16 \
  --epochs 2000 \
  --lr-hypernet 1e-3 --lr-embedder 5e-6 \
  --save-dir experiments \
  --experiment-name hypergp_cub_5way5shot
```

This will launch meta-training for 2,000 episodes, fine-tuning the embedder, and save all outputs under `experiments/hypergp_cub_5way5shot/`.
