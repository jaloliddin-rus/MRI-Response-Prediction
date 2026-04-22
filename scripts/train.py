# -*- coding: utf-8 -*-
"""
Created on 24/02/2024

@author: Zahiriddin & Jaloliddin
"""
# Utilities
import os
import sys
import numpy as np
import pandas as pd
import torch
import pickle
import time
import argparse
import psutil
from datetime import datetime

# Make the project root importable so `src.*` and `architectures.*` resolve
# regardless of where python is invoked from.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch.utils.data import Subset
from monai.data import DataLoader
from monai.transforms import Compose, ToTensor

# Model Import
from architectures.Regressor import CustomRegressor
from architectures.BasicUNet import BasicUNet
from architectures.AutoEncoder import AutoEncoder
from architectures.DenseNet import DenseNet169
from architectures.EfficientNet import EfficientNetBN

# Shared data + loss utilities
from src.dataset import TiffDataset, custom_collate
from src.losses import custom_l1_loss, signal_correlation

# Optimizer
import torch.optim as optim

# Loss Functions
from torch.nn import L1Loss, MSELoss, HuberLoss

# Plotting
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split



def parse_args():
    parser = argparse.ArgumentParser(description="Train a model on MRI data.")
    parser.add_argument('--architecture', type=str, default='Regressor',
                        help='Model architecture: Regressor')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate for the optimizer.')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of epochs to train.')
    parser.add_argument('--loss_fn', type=str, default='CustomL1Loss',
                        help='Loss function to use: L1Loss, MSELoss, CustomL1Loss')
    parser.add_argument('--dir', type=str, default='data',
                        help='Directory of Data (structured: data/animal/...)')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='Optimizer to use: Adam, SGD, RMSprop')
    return parser.parse_args()

def get_gpu_memory_usage():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**2  # Convert to MB

def reset_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def get_process_memory():
    return psutil.Process().memory_info().rss / 1024**2  # Convert to MB

def measure_memory_usage(func):
    def wrapper(*args, **kwargs):
        # Reset peak GPU memory stats
        reset_gpu_memory()
        
        # Measure RAM before
        ram_before = get_process_memory()
        
        # Run the function
        result = func(*args, **kwargs)
        
        # Measure RAM and GPU after
        ram_after = get_process_memory()
        gpu_memory = get_gpu_memory_usage()
        
        return result, ram_after - ram_before, gpu_memory
    
    return wrapper

@measure_memory_usage
def train_epoch(model, train_loader, optimizer, loss_function, device):
    model.train()
    epoch_loss = 0.0
    for batch in train_loader:
        images = batch['images'].to(device)
        mri_params = batch['mri_params'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(images, mri_params)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * images.size(0)
    
    return epoch_loss / len(train_loader.dataset)

def validate_model(model, val_loader, device, loss_function):
    model.eval()
    total_loss = 0.0
    total_corr = 0.0
    with torch.no_grad():
        for batch in val_loader:
            images = batch['images'].to(device)
            mri_params = batch['mri_params'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images, mri_params)
            loss = loss_function(outputs, labels)
            corr = signal_correlation(labels, outputs)
            
            total_loss += loss.item() * images.size(0)
            total_corr += corr.item() * images.size(0)

    avg_loss = total_loss / len(val_loader.dataset)
    avg_corr = total_corr / len(val_loader.dataset)
    return avg_loss, avg_corr

if __name__ == "__main__":
    print("is CUDA enabled?: ", torch.cuda.is_available())
    args = parse_args()
    data_dir = args.dir
    model = args.architecture
    
    os.makedirs('splits', exist_ok=True)
    data_list_path = os.path.join('splits', 'data_list.pkl')
    if os.path.exists(data_list_path):
        with open(data_list_path, 'rb') as f:
            data_list = pickle.load(f)
    else:
        data_list = []
        for animal_dir in os.listdir(data_dir):
            animal_path = os.path.join(data_dir, animal_dir)
            if os.path.isdir(animal_path):
                for chunk_dir in os.listdir(animal_path):
                    chunk_path = os.path.join(animal_path, chunk_dir)

                    tiff_files = [os.path.join(chunk_path, f) for f in os.listdir(chunk_path) if f.endswith('.tiff')]
                    npy_file = os.path.join(chunk_path, "signals_journal.npy")

                    if len(tiff_files) == 9 and os.path.exists(npy_file):
                        npy_data = np.load(npy_file)

                        # Check each field in the structured array
                        is_valid = True
                        for field in npy_data.dtype.names:
                            if np.issubdtype(npy_data[field].dtype, np.floating):
                                if np.isnan(npy_data[field]).any() or (npy_data[field] == 0).all():
                                    is_valid = False
                                    break
                            elif np.issubdtype(npy_data[field].dtype, np.integer):
                                if (npy_data[field] == 0).all():
                                    is_valid = False
                                    break

                        if is_valid:
                            data_list.append({"images": tiff_files, "label": npy_file})

        with open(data_list_path, 'wb') as f:
            pickle.dump(data_list, f)

    train_transforms = Compose([
        ToTensor()
    ])

    test_transforms = Compose([
        ToTensor()
    ])

    train_indices_path = os.path.join('splits', 'train_indices.pkl')
    val_indices_path = os.path.join('splits', 'val_indices.pkl')
    test_indices_path = os.path.join('splits', 'test_indices.pkl')

    if os.path.exists(train_indices_path) and os.path.exists(val_indices_path) and os.path.exists(test_indices_path):
        with open(train_indices_path, 'rb') as f:
            train_indices = pickle.load(f)
        with open(val_indices_path, 'rb') as f:
            val_indices = pickle.load(f)
        with open(test_indices_path, 'rb') as f:
            test_indices = pickle.load(f)
    else:
        indices = np.arange(len(data_list))
        train_indices, temp_indices = train_test_split(indices, test_size=0.2, random_state=42)
        val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)

        with open(train_indices_path, 'wb') as f:
            pickle.dump(train_indices.tolist(), f)
        with open(val_indices_path, 'wb') as f:
            pickle.dump(val_indices.tolist(), f)
        with open(test_indices_path, 'wb') as f:
            pickle.dump(test_indices.tolist(), f)

    train_dataset = Subset(TiffDataset(data_list, train_transforms), train_indices)
    val_dataset = Subset(TiffDataset(data_list, test_transforms), val_indices)
    test_dataset = Subset(TiffDataset(data_list, test_transforms), test_indices)

    num_train_samples = len(train_dataset)
    num_val_samples = len(val_dataset)
    num_test_samples = len(test_dataset)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=custom_collate, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate)

    print(f"Number of training samples: {num_train_samples}")
    print(f"Number of validation samples: {num_val_samples}")
    print(f"Number of testing samples: {num_test_samples}")

    #MODEL
    if model == "Regressor":
        model = CustomRegressor()
    elif model == "BasicUNet":
        model = BasicUNet(spatial_dims=3, in_channels=9, out_channels=11)
    elif model == "AutoEncoder":
        model = AutoEncoder(
            spatial_dims=3,
            in_channels=9,  # For 9 TIFF images
            out_channels=32,  # 32 channels in the bottleneck layer
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
            dropout_prob=0.2
        )
    elif model == "EfficientNetB4":
        model = EfficientNetBN(
            model_name="efficientnet-b4",
            spatial_dims=3,
            in_channels=9,
            num_classes=550,  # 11 signals × 50 time points
            pretrained=False  # Since we're using 3D input
        )


    #LOSS FUNCTION
    if args.loss_fn == "L1Loss":
        loss_function = L1Loss()
    elif args.loss_fn == "MSELoss":
        loss_function = MSELoss()
    elif args.loss_fn == "CustomL1Loss":
        loss_function = custom_l1_loss
    elif args.loss_fn == "HuberLoss":
        loss_function = HuberLoss()
    else:
        raise ValueError("Unknown loss function specified")

    #OPTIMIZER
    if args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    elif args.optimizer == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    else:
        raise ValueError("Unknown optimizer specified")
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    num_epochs = args.epochs
    train_losses = []
    val_losses = []
    val_correlations = []

    patience = 15
    best_val_metric = float('inf')
    epochs_no_improve = 0
    early_stop = False

    start_time = time.time()
    epoch_data_list = []
    resource_usage_list = []

    for epoch in range(num_epochs):
        epoch_time = time.time()
        if early_stop:
            print("Early stopping due to no improvement in validation metric!")
            break
        
        epoch_start_time = time.time()
        
         # Training
        avg_train_loss, ram_usage, gpu_memory = train_epoch(model, train_loader, optimizer, loss_function, device)
        train_losses.append(avg_train_loss)
        
        # Validation
        val_loss, val_corr = validate_model(model, val_loader, device, loss_function)
        val_losses.append(val_loss)
        val_correlations.append(val_corr)

        val_metric = val_loss - val_corr  # Lower is better

        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time

        print(f"Epoch {epoch+1}/{num_epochs}: Training Loss: {avg_train_loss:.4f}, Val. Loss: {val_loss:.4f}, "
              f"Val. Correlation: {val_corr:.4f}, Epoch Time: {epoch_time:.2f}s, "
              f"Training RAM Usage: {ram_usage:.2f}MB, Training GPU Memory: {gpu_memory:.2f}MB")
        
        epoch_data = {
            'Epoch': epoch + 1,
            'Training Loss': avg_train_loss,
            'Validation Loss': val_loss,
            'Validation Correlation': val_corr,
            'Epoch Time (s)': epoch_time
        }
        epoch_data_list.append(epoch_data)
        
        resource_usage = {
            'Epoch': epoch + 1,
            'Training RAM Usage (MB)': ram_usage,
            'Training GPU Memory (MB)': gpu_memory,
            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        resource_usage_list.append(resource_usage)

        if val_metric < best_val_metric:
            best_val_metric = val_metric
            epochs_no_improve = 0
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), f'models/best_{args.architecture}_{args.loss_fn}_{args.optimizer}.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print("Early stopping triggered!")
                early_stop = True

        scheduler.step()

    
    # Save the final model
    end_time = time.time()
    total_training_time = end_time - start_time
    # Calculate average memory usage
    avg_ram_usage = sum(resource['Training RAM Usage (MB)'] for resource in resource_usage_list) / len(resource_usage_list)
    avg_gpu_memory = sum(resource['Training GPU Memory (MB)'] for resource in resource_usage_list) / len(resource_usage_list)

    print(f"Training completed in: {total_training_time:.2f} seconds")
    print(f"Average RAM Usage: {avg_ram_usage:.2f} MB")
    print(f"Average GPU Memory Usage: {avg_gpu_memory:.2f} MB")

    # Save the averages and total time to a file
    os.makedirs('results', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    with open(f'results/training_summary_{args.architecture}_{args.loss_fn}_{args.optimizer}.txt', 'w') as f:
        f.write(f"Total Training Time: {total_training_time:.2f} seconds\n")
        f.write(f"Average RAM Usage: {avg_ram_usage:.2f} MB\n")
        f.write(f"Average GPU Memory Usage: {avg_gpu_memory:.2f} MB\n")

    # Update the resource usage DataFrame to include averages and total time
    resource_df = pd.DataFrame(resource_usage_list)
    resource_df['Total Training Time (s)'] = total_training_time
    resource_df['Average RAM Usage (MB)'] = avg_ram_usage
    resource_df['Average GPU Memory Usage (MB)'] = avg_gpu_memory

    # Save the updated resource usage data
    resource_df.to_csv(f'results/resource_usage_{args.architecture}_{args.loss_fn}_{args.optimizer}.csv', index=False)

    # Save training history
    history_df = pd.DataFrame(epoch_data_list)
    history_df['Total Training Time (s)'] = total_training_time
    history_df['Average RAM Usage (MB)'] = avg_ram_usage
    history_df['Average GPU Memory Usage (MB)'] = avg_gpu_memory
    history_df.to_csv(f'results/history_{args.architecture}_{args.loss_fn}_{args.optimizer}.csv', index=False)

    # Plotting training and validation metrics
    plt.figure(figsize=(12, 8))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.title('Training and Validation Metrics')
    plt.legend()

    plt.savefig(f'figures/{args.architecture}_{args.loss_fn}_{args.optimizer}_metrics.svg', format='svg')
    plt.savefig(f'figures/{args.architecture}_{args.loss_fn}_{args.optimizer}_metrics.png', format='png', dpi=300)
    plt.show()

    print("Training completed. Model, history, and resource usage data saved.")