import os
import sys
import numpy as np
import torch
import argparse
import pickle
from tqdm import tqdm
from monai.transforms import Compose, ToTensor
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import r2_score, explained_variance_score
import psutil
import time

# Make the project root importable so `src.*` and `architectures.*` resolve
# regardless of where python is invoked from.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from architectures.Regressor import CustomRegressor
from architectures.BasicUNet import BasicUNet
from architectures.AutoEncoder import AutoEncoder
from architectures.DenseNet import DenseNet169
from architectures.EfficientNet import EfficientNetBN
from src.dataset import TiffDataset, custom_collate


def parse_args():
    parser = argparse.ArgumentParser(description="Test a trained model on MRI data.")
    parser.add_argument('--architecture', type=str, default='Regressor',
                        help='Model architecture: Regressor, DenseNet169, BasicUNet, EfficientNetB4, AutoEncoder')
    parser.add_argument('--loss_fn', type=str, default='CustomL1Loss',
                        help='Loss function used: L1Loss, MSELoss, CustomL1Loss, HuberLoss')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='Optimizer used: Adam, AdamW')
    parser.add_argument('--dir', type=str, default='data',
                        help='Directory of Data (structured: data/animal/...)')
    return parser.parse_args()


def get_gpu_memory_usage():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2  # Convert to MB
    else:
        return 0

def get_ram_usage():
    return psutil.Process().memory_info().rss / 1024**2  # RSS in MB

def main():
    args = parse_args()
    model_name = f"best_{args.architecture}_{args.loss_fn}_{args.optimizer}"
    print(f"--Processing model: {model_name}")
    model = args.architecture

    # Load data_list
    with open(os.path.join('splits', 'data_list.pkl'), 'rb') as f:
        data_list = pickle.load(f)

    # Load test indices
    with open(os.path.join('splits', 'test_indices.pkl'), 'rb') as f:
        test_indices = pickle.load(f)

    # Use the same transforms as in training
    test_transforms = Compose([
        ToTensor()
    ])

    # Create test_ds using the same transforms
    test_ds = TiffDataset(data_list, transform=test_transforms)

    # Create the test_dataset using the test indices
    test_dataset = Subset(test_ds, test_indices)

    # Create DataLoader with the same parameters
    batch_size = 32
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate, pin_memory=True)

    if model == "Regressor":
        model = CustomRegressor()
    elif model == "BasicUNet":
        model = BasicUNet(spatial_dims=3, in_channels=9, out_channels=11)
    elif model == "AutoEncoder":
        model = AutoEncoder(
            spatial_dims=3,
            in_channels=9,  # For 9 TIFF images
            out_channels=32,  # Or any other suitable number
            channels=(32, 64, 128, 256),
            strides=(2, 2, 2, 2)
    )
    elif model == "DenseNet169":
        model = DenseNet169(
            spatial_dims=3,          # 3D input images
            in_channels=9,          # 9 TIFF images
            out_channels=11,        # 11 signals
            init_features=64,
            growth_rate=32,
            dropout_prob=0.2        # Adjust as needed
        )
    elif model == "EfficientNetB4":
        model = EfficientNetBN(
            model_name="efficientnet-b4",
            spatial_dims=3,
            in_channels=9,
            num_classes=550,  # 11 signals × 50 time points
            pretrained=False  # Since we're using 3D input
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(f'models/{model_name}.pth', map_location=device))
    model.eval()
    model = model.to(device)

    mse_values = []
    mae_values = []
    r2_values = []
    all_predictions = []
    all_labels = []
    test_data_list = []
    prediction_times = []
    gpu_memory_usage = []
    ram_usage = []

    total_samples = 0
    total_time = 0

    for batch_idx, data in enumerate(tqdm(test_loader, desc="Testing")):
        images = data['images'].to(device)
        mri_params = data['mri_params'].to(device)
        labels = data['label'].to(device)
        original_params = data['original_params'].cpu().numpy()

        batch_start_time = time.time()
        with torch.no_grad():
            predictions = model(images, mri_params).cpu().numpy()
        batch_end_time = time.time()

        batch_time = batch_end_time - batch_start_time
        batch_size_actual = predictions.shape[0]
        total_samples += batch_size_actual
        total_time += batch_time

        all_predictions.append(predictions)
        all_labels.append(labels.cpu().numpy())

        #batch_size_actual = predictions.shape[0]
        for i in range(batch_size_actual):
            mse = ((predictions[i] - labels[i].cpu().numpy()) ** 2).mean()
            mae = np.abs(predictions[i] - labels[i].cpu().numpy()).mean()
            r2 = r2_score(labels[i].cpu().numpy().flatten(), predictions[i].flatten())

            mse_values.append(mse)
            mae_values.append(mae)
            r2_values.append(r2)
            prediction_times.append(batch_time / batch_size_actual)

            batch_data = {
                'MSE': mse,
                'MAE': mae,
                'R2': r2,
                'b_value': original_params[i, 0],
                'small_delta': original_params[i, 1],
                'big_delta': original_params[i, 2],
                'Prediction Time (s)': batch_time / batch_size_actual
            }
            test_data_list.append(batch_data)
        
        gpu_memory_usage.append(get_gpu_memory_usage())
        ram_usage.append(get_ram_usage())
    
    average_prediction_time = total_time / total_samples
    average_gpu_memory = np.mean(gpu_memory_usage)
    average_ram_usage = np.mean(ram_usage)

    # Calculate aggregate statistics
    all_predictions = np.concatenate(all_predictions, axis=0)  # Shape: (n_samples, 11, 50)
    all_labels = np.concatenate(all_labels, axis=0)  # Shape: (n_samples, 11, 50)

    # Compute R2 score and explained variance
    y_true = all_labels.flatten()
    y_pred = all_predictions.flatten()
    overall_r2 = r2_score(y_true, y_pred)
    exp_var = explained_variance_score(y_true, y_pred)

    # Calculate percentage of samples with R² >= 80%
    r2_threshold = 0.80
    samples_above_threshold = sum(r2 >= r2_threshold for r2 in r2_values)
    percentage_above_threshold = (samples_above_threshold / len(r2_values)) * 100

    mse = np.mean(mse_values)
    mae = np.mean(mae_values)
    std_mse = np.std(mse_values)
    std_mae = np.std(mae_values)
    min_mse = np.min(mse_values)
    max_mse = np.max(mse_values)
    min_mae = np.min(mae_values)
    max_mae = np.max(mae_values)
    median_mse = np.median(mse_values)
    median_mae = np.median(mae_values)
    rmse = np.sqrt(mse)
    
    print(f"Results for: {model_name}")
    print(f"Overall R2 Score: {overall_r2:.8f}")
    print(f"Explained Variance: {exp_var:.8f}")
    print(f"Percentage of samples with R² >= 80%: {percentage_above_threshold:.2f}%")
    print(f"Average prediction time per sample: {average_prediction_time:.6f} seconds")
    print(f"Average GPU memory usage: {average_gpu_memory:.2f} MB")
    print(f"Average RAM usage: {average_ram_usage:.2f} MB")

    # Create directories for saving results
    os.makedirs("results/mse", exist_ok=True)
    os.makedirs("results/mae", exist_ok=True)
    os.makedirs("results/metrics", exist_ok=True)
    #os.makedirs("results/data", exist_ok=True)

    # Save MSE and MAE values
    with open(f"results/mse/mse_{model_name}.pkl", "wb") as f:
        pickle.dump(mse_values, f)

    with open(f"results/mae/mae_{model_name}.pkl", "wb") as f:
        pickle.dump(mae_values, f)

    # Save metrics to text file
    with open(f"results/metrics/{model_name}.txt", "w") as f:
        f.write(f"Architecture: {args.architecture}\n")
        f.write(f"Loss Function: {args.loss_fn}\n")
        f.write(f"Optimizer: {args.optimizer}\n")
        f.write(f"MSE: {mse:.8f}\n")
        f.write(f"MAE: {mae:.8f}\n")
        f.write(f"SD of MSE: {std_mse:.8f}\n")
        f.write(f"SD of MAE: {std_mae:.8f}\n")
        f.write(f"Min MSE: {min_mse:.8f}\n")
        f.write(f"Max MSE: {max_mse:.8f}\n")
        f.write(f"Median MSE: {median_mse:.8f}\n")
        f.write(f"Min MAE: {min_mae:.8f}\n")
        f.write(f"Max MAE: {max_mae:.8f}\n")
        f.write(f"Median MAE: {median_mae:.8f}\n")
        f.write(f"Overall R2 Score: {overall_r2:.8f}\n")
        f.write(f"Explained Variance: {exp_var:.8f}\n")
        f.write(f"RMSE: {rmse:.8f}\n")
        f.write(f"Percentage of samples with R² >= 80%: {percentage_above_threshold:.2f}%\n")
        f.write(f"Average Prediction Time per Sample: {average_prediction_time:.6f} seconds\n")
        f.write(f"Average GPU Memory Usage: {average_gpu_memory:.2f} MB\n")
        f.write(f"Average RAM Usage: {average_ram_usage:.2f} MB\n")

    # Save test data list
    #test_df = pd.DataFrame(test_data_list)
    #test_df.to_csv(f"results/data/test_data_{model_name}.csv", index=False)

    print("Testing complete!")

if __name__ == "__main__":
    main()