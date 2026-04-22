"""
Time-Series Prediction Visualization for MRI Signal Prediction Models

This script visualizes model predictions for specific test samples, showing the full
50-point time-series signal for each of the 11 phi/theta gradient direction combinations.

Main functionality:
- Loads a trained model (any architecture) and a specific test sample
- Generates predictions for 11 angle combinations (φ, θ)
- For each angle, plots the full 50-point time-series:
  * Ground truth signal (blue solid line)
  * Predicted signal (red dashed line)
  * Readout time marker (green vertical line + dot)
- Calculates and displays per-angle metrics (MSE, MAE, R²)
- Creates a publication-ready 4×3 grid figure

Key feature - Readout Time:
The readout time is automatically detected as the peak after the initial signal dip,
marked with a green vertical line. This is a critical MRI timing parameter.

Use case:
Examine how well a model predicts the temporal dynamics of MRI signals across
different gradient directions for a specific vascular structure.

Usage:
    # Visualize a specific test sample
    python plot.py --architecture DenseNet169 --loss_fn HuberLoss --optimizer Adam --index 239
    
    # Filter by specific MRI parameters
    python plot.py --architecture DenseNet169 --loss_fn HuberLoss --optimizer Adam --index 239 --b_value 500 --small_delta 1.0 --big_delta 4.0

Arguments:
    --architecture: Model architecture (DenseNet169, Regressor, BasicUNet, AutoEncoder, EfficientNetB4)
    --loss_fn: Loss function used in training (HuberLoss, MSELoss, L1Loss, etc.)
    --optimizer: Optimizer used in training (Adam, AdamW)
    --index: Test sample index to visualize
    --b_value: (Optional) Filter signals by b-value
    --small_delta: (Optional) Filter by small delta (δ)
    --big_delta: (Optional) Filter by big delta (Δ)
    --dir: Data directory (default: 'data')

Output:
    Saves a high-resolution figure to output/plots/ showing all 11 angle predictions
    with time-series, readout times, and per-angle performance metrics.
"""

import os
import numpy as np
import torch
import argparse
import pickle
from monai.transforms import Compose, ToTensor
from torch.utils.data import Subset
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except (AttributeError, OSError):
    pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from architectures.Regressor import CustomRegressor
from architectures.BasicUNet import BasicUNet
from architectures.AutoEncoder import AutoEncoder
from architectures.DenseNet import DenseNet169
from architectures.EfficientNet import EfficientNetBN
from src.dataset import TiffDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="Plot predictions from a trained model on MRI data.")
    parser.add_argument('--architecture', type=str, default='Regressor',
                        help='Model architecture: Regressor, RegressorV2, DenseNet169, DenseNet201, BasicUNet, EfficientNetB3, AutoEncoder')
    parser.add_argument('--loss_fn', type=str, default='CustomL1Loss',
                        help='Loss function used: L1Loss, MSELoss, CustomL1Loss, HuberLoss')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='Optimizer used: Adam, SGD, RMSprop, AdamW')
    parser.add_argument('--dir', type=str, default='data',
                        help='Directory of Data (structured: data/animal/...)')
    parser.add_argument('--index', type=int, required=True,
                        help='Index of the sample to plot')
    parser.add_argument('--b_value', type=float, help='b value to filter the signals')
    parser.add_argument('--small_delta', type=float, help='Small delta value to filter the signals')
    parser.add_argument('--big_delta', type=float, help='Big delta value to filter the signals')
    return parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(ground_truth, prediction):
    mse = mean_squared_error(ground_truth, prediction)
    mae = mean_absolute_error(ground_truth, prediction)
    r2 = r2_score(ground_truth, prediction)
    return mse, mae, r2

def find_readout_time(signal):
    """
    Find the readout time by identifying the peak after the initial dip.
    """
    # Find the first local minimum
    min_idx = None
    for i in range(1, len(signal)-1):
        if signal[i-1] > signal[i] < signal[i+1]:
            min_idx = i
            break
    
    if min_idx is None or min_idx >= len(signal)-2:
        # If no clear dip is found, use midpoint as approximation
        return len(signal) // 2
    
    # Find the highest point after the dip
    readout_idx = np.argmax(signal[min_idx:]) + min_idx
    return readout_idx

def plot_and_save_sample(model_file, sample, sample_path, args, animal_chunk=None):
    model_name = os.path.basename(model_file).split(".")[0]
    model_parts = model_name.split("_")
    architecture, loss_func, optimizer = model_parts[1], model_parts[2], model_parts[3]

    if architecture == "Regressor":
        model = CustomRegressor()
    elif architecture == "BasicUNet":
        model = BasicUNet(spatial_dims=3, in_channels=9, out_channels=11)
    elif architecture == "AutoEncoder":
        model = AutoEncoder(
            spatial_dims=3,
            in_channels=9,
            out_channels=32,
            channels=(32, 64, 128, 256),
            strides=(2, 2, 2, 2)
        )
    elif architecture == "DenseNet169":
        model = DenseNet169(
            spatial_dims=3,
            in_channels=9,
            out_channels=11,
            init_features=64,
            growth_rate=32,
            dropout_prob=0.2
        )
    elif architecture == "EfficientNetB4":
        model = EfficientNetBN(
            model_name="efficientnet-b4",
            spatial_dims=3,
            in_channels=9,
            num_classes=550,
            pretrained=False
        )
    
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()
    model = model.to(device)

    images, mri_params, ground_truth, original_params = sample['images'], sample['mri_params'], sample['label'], sample['original_params']

    images_batch = images.unsqueeze(0).to(device)
    mri_params_batch = mri_params.unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(images_batch, mri_params_batch).cpu().numpy()[0]

    ground_truth = ground_truth.numpy()

    overall_metrics = compute_metrics(ground_truth.flatten(), predictions.flatten())
    print(f"Overall metrics: MSE={overall_metrics[0]:.4f}, MAE={overall_metrics[1]:.4f}, R²={overall_metrics[2]:.2f}")

    metrics = [compute_metrics(gt, pred) for gt, pred in zip(ground_truth, predictions)]

    # Create figure with 4x3 subplots (slightly wider to maximize panel area)
    fig, axes = plt.subplots(4, 3, figsize=(11, 14))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.98, bottom=0.06,
                        wspace=0.25, hspace=0.30)
    
    phi_theta_list = [
        (0, 0), (0, 90), (45, 45), (45, 90), (45, 135),
        (90, 45), (90, 90), (90, 135), (135, 45), (135, 90), (135, 135),
    ]

    # Plot each signal
    for i in range(11):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        phi, theta = phi_theta_list[i]
        
        # Plot ground truth and prediction
        ax.plot(range(50), ground_truth[i], label='Ground Truth', linestyle='-', color='blue', linewidth=2)
        ax.plot(range(50), predictions[i], label='Prediction', linestyle='--', color='red', linewidth=3)

        # Find readout time for ground truth
        readout_idx = find_readout_time(ground_truth[i])
        
        # Plot vertical line at readout time
        ax.axvline(x=readout_idx, color='green', linestyle='-.', linewidth=1.5, 
                   label=f'Readout Time')
        
        # Highlight the readout point
        ax.plot(readout_idx, ground_truth[i][readout_idx], 'go', markersize=8)

        mse, mae, r2 = metrics[i]
        # if phi == 0 and theta == 0:
        #     ax.set_title(f"$\\phi={phi}^\\circ$, $\\theta={theta}^\\circ$ (Spin Echo)\nMSE={mse:.4f}, MAE={mae:.4f}, $R^2$={r2:.2f}", fontsize=10)
        # else: 
        #     ax.set_title(f"$\\phi={phi}^\\circ$, $\\theta={theta}^\\circ$\nMSE={mse:.4f}, MAE={mae:.4f}, $R^2$={r2:.2f}", fontsize=10)

        # Subplot titles retained; no main figure title
        if phi == 0 and theta == 0:
            ax.set_title(f"$\\phi={phi}^\\circ$, $\\theta={theta}^\\circ$ (Spin Echo)", fontsize=10)
        else:
            ax.set_title(f"$\\phi={phi}^\\circ$, $\\theta={theta}^\\circ$", fontsize=10)

        metrics_text = f"MSE={mse:.4f}, MAE={mae:.4f}, $R^2$={r2:.2f}"
        ax.text(0.98, 0.02, metrics_text, transform=ax.transAxes, fontsize=9,
        horizontalalignment='right', verticalalignment='bottom',
        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=3))
        
        # Only show y-axis label for the first plot in each row
        if col == 0:
            ax.set_ylabel('Signal')
        else:
            ax.set_ylabel('')
            
        ax.set_xlabel('Time')
        ax.set_ylim(0.0, 1.0)
        ax.set_xlim(0, 49)
        ax.legend(fontsize=7, loc='upper right', framealpha=0.6)

        print(f"-φ={phi}°, θ={theta}°:")
        print(f"--MSE={mse:.4f}, MAE={mae:.4f}, R²={r2:.4f}, Readout Time={readout_idx}")

    # Remove the last subplot if not used
    if axes.size > 11:
        fig.delaxes(axes[3, 2])

    b_value, small_delta, big_delta = original_params[0].item(), original_params[1].item(), original_params[2].item()

    # Keep figures title-free for publication; layout already adjusted above
    plt.tight_layout()

    save_path = "results/plots/predictions"
    os.makedirs(save_path, exist_ok=True)
    safe_chunk = (animal_chunk if animal_chunk else sample_path).replace("/", "_")
    filename = f'prediction_{architecture}_{loss_func}_{optimizer}_{safe_chunk}_b{b_value}_d{small_delta}_D{big_delta}.pdf'
    plt.savefig(os.path.join(save_path, filename), format='pdf', bbox_inches='tight')
    plt.close()

    avg_mse = np.mean([m[0] for m in metrics])
    avg_mae = np.mean([m[1] for m in metrics])
    avg_r2 = np.nanmean([m[2] for m in metrics])
    print(f"Average metrics: MSE={avg_mse:.4f}, MAE={avg_mae:.4f}, R²={avg_r2:.4f}")

def main():
    set_seed(42)  # Set a seed for reproducibility
    args = parse_args()
    model_name = f"best_{args.architecture}_{args.loss_fn}_{args.optimizer}"
    models_path = f"models/{model_name}.pth"

    print(f"Loading model: {models_path}")

    # Load data_list
    with open(os.path.join('splits', 'data_list.pkl'), 'rb') as f:
        data_list = pickle.load(f)

    # Load test indices
    with open(os.path.join('splits', 'test_indices.pkl'), 'rb') as f:
        test_indices = pickle.load(f)

    # Use the same transforms as in training
    test_transforms = Compose([ToTensor()])

    # Create test_ds using the same transforms and specified MRI parameters (if any)
    test_ds = TiffDataset(
        data_list, transform=test_transforms,
        b_value=args.b_value, small_delta=args.small_delta, big_delta=args.big_delta,
        include_animal_chunk=True,
    )

    # Create the test_dataset using the test indices
    test_dataset = Subset(test_ds, test_indices)

    if len(test_dataset) == 0:
        print("No data found for the specified parameters.")
        return

    if args.index >= len(test_dataset):
        print(f"Error: Index {args.index} is out of range. Max index is {len(test_dataset) - 1}")
        return

    # Get the sample - it already contains the correct animal_chunk from TiffDataset
    sample = test_dataset[args.index]
    
    # Get animal_chunk from sample (already extracted correctly in TiffDataset)
    animal_chunk = sample.get('animal_chunk', 'unknown/unknown') if isinstance(sample, dict) else 'unknown/unknown'
    sample_path = animal_chunk  # For backwards compatibility
    
    print(f"Plotting sample at index {args.index}")
    print(f"Sample: {animal_chunk}")
    print(f"MRI Parameters: b={sample['original_params'][0]:.1f}, δ={sample['original_params'][1]:.1f}, Δ={sample['original_params'][2]:.1f}")
    
    plot_and_save_sample(models_path, sample, sample_path, args, animal_chunk)
    print("Plot saved.")

if __name__ == "__main__":
    main()