import os
import numpy as np
import torch
import argparse
import pickle
from monai.transforms import Compose, ToTensor
from torch.utils.data import Dataset, Subset
import SimpleITK as sitk
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from architectures.Regressor import CustomRegressor
from architectures.BasicUNet import BasicUNet
from architectures.AutoEncoder import AutoEncoder
from architectures.DenseNet import DenseNet169
from architectures.EfficientNet import EfficientNetBN
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="Plot combined predictions from a trained model on MRI data.")
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


class TiffDataset(Dataset):
    def __init__(self, data, b_value=None, small_delta=None, big_delta=None, transform=None):
        self.data = []
        self.transform = transform

        # Define normalization ranges
        self.b_range = [50, 500]
        self.small_delta_range = [1, 2]
        self.big_delta_range = [4, 7]

        # Predefined phi and theta combinations
        self.phi_theta_list = [
            (0, 0), (0, 90), (45, 45), (45, 90), (45, 135),
            (90, 45), (90, 90), (90, 135), (135, 45), (135, 90), (135, 135),
        ]

        for item in data:
            tiff_files = item['images']
            npy_file = item['label']
            npy_data = np.load(npy_file, allow_pickle=True)

            if npy_data.ndim == 0:
                npy_data = npy_data.item()

            # Filter data if MRI parameters are specified
            if b_value is not None and small_delta is not None and big_delta is not None:
                filtered_data = [row for row in npy_data if row[1] == b_value and row[2] == small_delta and row[3] == big_delta]
            else:
                filtered_data = npy_data

            # Group data by b_value, small_delta, and big_delta
            grouped_data = {}
            for row in filtered_data:
                row_tuple = tuple(row.item())
                key = (row_tuple[1], row_tuple[2], row_tuple[3])  # (b_value, small_delta, big_delta)
                if key not in grouped_data:
                    grouped_data[key] = []
                grouped_data[key].append(row_tuple)

            for (b, small_delta, big_delta), group in grouped_data.items():
                signals = []
                for phi, theta in self.phi_theta_list:
                    found = False
                    for row in group:
                        if row[4] == phi and row[5] == theta:
                            signals.append(row[0])
                            found = True
                            break
                    if not found:
                        break
                else:
                    # All phi-theta pairs were found
                    signals = np.array(signals, dtype=np.float32)

                    # Normalize parameters
                    norm_b = (b - self.b_range[0]) / (self.b_range[1] - self.b_range[0])
                    norm_small_delta = (small_delta - self.small_delta_range[0]) / (self.small_delta_range[1] - self.small_delta_range[0])
                    norm_big_delta = (big_delta - self.big_delta_range[0]) / (self.big_delta_range[1] - self.big_delta_range[0])

                    # Store MRI parameters as scalar values
                    mri_params = np.array([norm_b, norm_small_delta, norm_big_delta], dtype=np.float32)

                    self.data.append({
                        'images': tiff_files,
                        'signals': signals,
                        'mri_params': mri_params,
                        'original_params': np.array([b, small_delta, big_delta], dtype=np.float32)
                    })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        if not hasattr(self, 'cached_images'):
            self.cached_images = {}
        image_key = tuple(item['images'])
        if image_key not in self.cached_images:
            volumes = [self.load_3d_tiff(img) for img in item['images']]
            if self.transform:
                volumes = [self.transform(vol) for vol in volumes]
            volume_4d = np.stack(volumes, axis=0).astype(np.float32)
            self.cached_images[image_key] = volume_4d
        else:
            volume_4d = self.cached_images[image_key]

        volume_4d = torch.from_numpy(volume_4d)
        signals = torch.from_numpy(item['signals']).float()
        mri_params = torch.from_numpy(item['mri_params']).float()
        original_params = torch.from_numpy(item['original_params']).float()

        return {
            "images": volume_4d,
            "mri_params": mri_params,
            "label": signals,
            "original_params": original_params
        }

    @staticmethod
    def load_3d_tiff(tiff_path):
        image = sitk.ReadImage(tiff_path)
        return sitk.GetArrayFromImage(image)

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

def plot_and_save_combined(model_file, sample, sample_path, args):
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

    phi_theta_list = [
        (0, 0), (0, 90), (45, 45), (45, 90), (45, 135),
        (90, 45), (90, 90), (90, 135), (135, 45), (135, 90), (135, 135),
    ]

    # Define a color palette for different phi/theta combinations
    colors = plt.cm.tab20(np.linspace(0, 1, 11))
    
    b_value, small_delta, big_delta = original_params[0].item(), original_params[1].item(), original_params[2].item()
    
    # ==================== Plot 1: Ground Truth ====================
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    
    for i in range(11):
        phi, theta = phi_theta_list[i]
        if phi == 0 and theta == 0:
            label = f'φ={phi}°, θ={theta}° (Spin Echo)'
        else:
            label = f'φ={phi}°, θ={theta}°'
        
        ax1.plot(range(50), ground_truth[i], label=label, color=colors[i], linewidth=2, alpha=0.8, linestyle='-')
    
    ax1.set_xlabel('Time', fontsize=14)
    ax1.set_ylabel('Signal', fontsize=14)
    ax1.set_ylim(0.0, 1.0)
    ax1.set_xlim(0, 49)
    ax1.legend(loc='best', fontsize=10, ncol=2)
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = "results/plots/predictions"
    os.makedirs(save_path, exist_ok=True)
    filename_base = f'groundtruth_{architecture}_{loss_func}_{optimizer}_{sample_path.replace("/", "_")}_b{b_value}_d{small_delta}_D{big_delta}'
    
    # Save as PDF
    plt.savefig(os.path.join(save_path, filename_base + '.pdf'), format='pdf', bbox_inches='tight')
    # Save as SVG
    plt.savefig(os.path.join(save_path, filename_base + '.svg'), format='svg', bbox_inches='tight')
    print(f"Ground truth plot saved: {filename_base}.pdf and {filename_base}.svg")
    plt.close()
    
    # ==================== Plot 2: Predictions ====================
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    
    for i in range(11):
        phi, theta = phi_theta_list[i]
        mse, mae, r2 = metrics[i]
        
        if phi == 0 and theta == 0:
            label = f'φ={phi}°, θ={theta}° (Spin Echo)'
        else:
            label = f'φ={phi}°, θ={theta}°'
        
        ax2.plot(range(50), predictions[i], label=label, color=colors[i], linewidth=2, alpha=0.8, linestyle='--')
        
        print(f"φ={phi}°, θ={theta}°: MSE={mse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
    
    ax2.set_xlabel('Time', fontsize=14)
    ax2.set_ylabel('Signal', fontsize=14)
    ax2.set_ylim(0.0, 1.0)
    ax2.set_xlim(0, 49)
    ax2.legend(loc='best', fontsize=10, ncol=2)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filename_base = f'predictions_{architecture}_{loss_func}_{optimizer}_{sample_path.replace("/", "_")}_b{b_value}_d{small_delta}_D{big_delta}'
    
    # Save as PDF
    plt.savefig(os.path.join(save_path, filename_base + '.pdf'), format='pdf', bbox_inches='tight')
    # Save as SVG
    plt.savefig(os.path.join(save_path, filename_base + '.svg'), format='svg', bbox_inches='tight')
    print(f"Predictions plot saved: {filename_base}.pdf and {filename_base}.svg")
    plt.close()
    
    # Print average metrics
    avg_mse = np.mean([m[0] for m in metrics])
    avg_mae = np.mean([m[1] for m in metrics])
    avg_r2 = np.nanmean([m[2] for m in metrics])
    print(f"\nAverage metrics across all directions: MSE={avg_mse:.4f}, MAE={avg_mae:.4f}, R²={avg_r2:.4f}")

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
    test_ds = TiffDataset(data_list, args.b_value, args.small_delta, args.big_delta, transform=test_transforms)

    # Create the test_dataset using the test indices
    test_dataset = Subset(test_ds, test_indices)

    if len(test_dataset) == 0:
        print("No data found for the specified parameters.")
        return

    if args.index >= len(test_dataset):
        print(f"Error: Index {args.index} is out of range. Max index is {len(test_dataset) - 1}")
        return

    # Get the sample and its path
    sample = test_dataset[args.index]
    original_idx = test_indices[args.index]
    sample_path = '/'.join(data_list[original_idx]['images'][0].split(os.sep)[-3:-1])
    
    print(f"Plotting sample at index {args.index}, sample path: {sample_path}")
    plot_and_save_combined(models_path, sample, sample_path, args)
    print("Plots saved successfully.")

if __name__ == "__main__":
    main()
