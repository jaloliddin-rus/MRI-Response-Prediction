"""
Common Good Sample Finder for MRI Signal Prediction Models

This script identifies test samples where ALL trained neural network models perform well,
making them ideal candidates for visualization and comparison in research papers/presentations.

Main functionality:
- Loads multiple trained models (Regressor, DenseNet169, etc.) with specified optimizer/loss
- Evaluates each model on the entire test dataset
- Identifies samples where ALL models achieve R² above a minimum threshold (default: 0.95)
- Ranks these "common good samples" by average R² across all models
- Outputs a list of top N samples with their MRI parameters and per-model R² scores
- Generates R² distribution plots for each model
- Provides ready-to-run commands for visualizing the identified samples

Use case:
When you need to show representative predictions in a paper, use this script to find
samples that consistently perform well across all architectures, demonstrating that
the prediction quality isn't architecture-specific.

Usage:
    python sample_finder.py --optimizer Adam --loss_fn HuberLoss --top_n 10 --min_r2 0.95

Arguments:
    --optimizer: Optimizer used in training (Adam, AdamW)
    --loss_fn: Loss function used (L1Loss, MSELoss, HuberLoss, etc.)
    --top_n: Number of top samples to identify (default: 10)
    --min_r2: Minimum R² threshold for a sample to be considered "good" (default: 0.95)
    --dir: Data directory (default: 'data')

Output files in results/common_samples/:
    - top_common_samples_{loss_fn}_{optimizer}.txt (detailed report)
    - top_common_samples_{loss_fn}_{optimizer}.csv (tabular data)
    - r2_distribution.png (histogram of R² scores per model)
"""

import os
import numpy as np
import torch
import argparse
import pickle
from tqdm import tqdm
from monai.transforms import Compose, ToTensor
from torch.utils.data import Subset, DataLoader
from sklearn.metrics import r2_score
import pandas as pd
import matplotlib.pyplot as plt
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
from src.dataset import TiffDataset, custom_collate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="Find samples with good performance across all models")
    parser.add_argument('--optimizer', type=str, default='Adam', 
                       help='Optimizer used: Adam, AdamW')
    parser.add_argument('--loss_fn', type=str, default='HuberLoss',
                       help='Loss function used: L1Loss, MSELoss, CustomL1Loss, HuberLoss')
    parser.add_argument('--models', type=str, nargs='+', 
                       default=['Regressor', 'DenseNet169'],
                       help='Models to evaluate: Regressor, BasicUNet, AutoEncoder, DenseNet169, EfficientNetB4')
    parser.add_argument('--top_n', type=int, default=10,
                       help='Number of top samples to identify')
    parser.add_argument('--min_r2', type=float, default=0.95,
                       help='Minimum acceptable R² score')
    parser.add_argument('--dir', type=str, default='data',
                       help='Directory of data')
    return parser.parse_args()

def get_model(architecture, optimizer, loss_fn):
    model_name = f"best_{architecture}_{loss_fn}_{optimizer}"
    model_path = f"models/{model_name}.pth"
    
    if not os.path.exists(model_path):
        return None
    
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
    else:
        return None
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model = model.to(device)
    return model

def evaluate_models(models, test_loader, min_r2):
    results = []
    
    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating samples")):
        images = batch['images'].to(device)
        mri_params = batch['mri_params'].to(device)
        labels = batch['label']
        
        batch_results = []
        
        for model_name, model in models.items():
            with torch.no_grad():
                predictions = model(images, mri_params).cpu()
            
            batch_size = predictions.shape[0]
            for i in range(batch_size):
                r2 = r2_score(labels[i].numpy().flatten(), predictions[i].numpy().flatten())
                if r2 >= min_r2:
                    sample_idx = batch_idx * test_loader.batch_size + i
                    batch_results.append({
                        'model': model_name,
                        'sample_idx': sample_idx,
                        'r2': r2,
                        'original_params': batch['original_params'][i].numpy()
                    })
        
        results.extend(batch_results)
    
    return results

def find_common_good_samples(results, architectures, top_n, min_r2):
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(results)
    
    # Group by sample_idx and count models with good performance
    sample_counts = df.groupby('sample_idx')['model'].nunique()
    
    # Find samples where all models perform well
    good_samples = sample_counts[sample_counts == len(architectures)].index.tolist()
    
    if not good_samples:
        print(f"No samples found where all models achieve R² >= {min_r2}")
        # Try lowering the requirement
        max_models = sample_counts.max()
        if max_models < len(architectures):
            print(f"The maximum number of models performing well on any sample is {max_models}/{len(architectures)}")
            good_samples = sample_counts[sample_counts == max_models].index.tolist()
    
    # For each good sample, calculate the average R² across all models
    sample_avg_r2 = {}
    for sample_idx in good_samples:
        sample_data = df[df['sample_idx'] == sample_idx]
        avg_r2 = sample_data['r2'].mean()
        min_r2_value = sample_data['r2'].min()
        sample_avg_r2[sample_idx] = (avg_r2, min_r2_value)
    
    # Sort by average R² (descending)
    sorted_samples = sorted(sample_avg_r2.items(), key=lambda x: x[1][0], reverse=True)
    
    # Take the top N samples
    top_samples = sorted_samples[:top_n]
    
    # Gather additional info about the top samples
    detailed_results = []
    for sample_idx, (avg_r2, min_r2_value) in top_samples:
        sample_data = df[df['sample_idx'] == sample_idx]
        model_r2s = {row['model']: row['r2'] for _, row in sample_data.iterrows()}
        # All rows should have the same original_params for a given sample_idx
        original_params = sample_data.iloc[0]['original_params']
        
        detailed_results.append({
            'sample_idx': sample_idx,
            'avg_r2': avg_r2,
            'min_r2': min_r2_value,
            'model_r2s': model_r2s,
            'b_value': original_params[0],
            'small_delta': original_params[1],
            'big_delta': original_params[2]
        })
    
    return detailed_results

def plot_r2_distribution(results, architectures, output_dir="results/analysis"):
    """Plot R² distribution for each model and highlight common good samples"""
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.DataFrame(results)
    
    # Create the figure
    plt.figure(figsize=(12, 8))
    
    # Plot R² distribution for each model
    for i, architecture in enumerate(architectures):
        model_data = df[df['model'] == architecture]
        plt.hist(model_data['r2'], bins=20, alpha=0.5, label=architecture)
    
    plt.xlabel('R² Score')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(f"{output_dir}/r2_distribution.png", dpi=300)
    plt.close()

def main():
    args = parse_args()
    
    # Load data_list and test indices
    with open(os.path.join('splits', 'data_list.pkl'), 'rb') as f:
        data_list = pickle.load(f)
    with open(os.path.join('splits', 'test_indices.pkl'), 'rb') as f:
        test_indices = pickle.load(f)
    
    # Use architectures from command line arguments
    architectures = args.models
    
    # Load models
    models = {}
    for architecture in architectures:
        model = get_model(architecture, args.optimizer, args.loss_fn)
        if model is not None:
            models[architecture] = model
            print(f"Loaded model: {architecture}")
        else:
            print(f"Model not found: {architecture}")
    
    if not models:
        print("No models found. Exiting.")
        return
    
    # Create test dataset
    test_transforms = Compose([ToTensor()])
    test_ds = TiffDataset(data_list, transform=test_transforms)
    test_dataset = Subset(test_ds, test_indices)
    
    # Create test dataloader
    batch_size = 8  # Use smaller batch size to avoid OOM
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=custom_collate
    )
    
    # Evaluate all models
    print(f"Evaluating {len(models)} models on {len(test_dataset)} test samples...")
    results = evaluate_models(models, test_loader, args.min_r2)
    
    # Find samples where all models perform well
    detailed_results = find_common_good_samples(
        results, 
        list(models.keys()), 
        args.top_n, 
        args.min_r2
    )
    
    if not detailed_results:
        print("No common good samples found. Try lowering the min_r2 threshold.")
        return
    
    # Save results to file
    output_dir = "results/common_samples"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as text file
    with open(f"{output_dir}/top_common_samples_{args.loss_fn}_{args.optimizer}.txt", "w") as f:
        f.write(f"Top {len(detailed_results)} Samples with Good Performance Across All Models\n")
        f.write(f"Minimum R² threshold: {args.min_r2}\n\n")
        
        for i, result in enumerate(detailed_results):
            f.write(f"Sample {i+1}:\n")
            f.write(f"  Test Index: {result['sample_idx']}\n")
            f.write(f"  Average R²: {result['avg_r2']:.4f}\n")
            f.write(f"  Minimum R²: {result['min_r2']:.4f}\n")
            f.write(f"  MRI Parameters: b={result['b_value']}, s_delta={result['small_delta']}, b_delta={result['big_delta']}\n")
            f.write("  Model Performance:\n")
            
            for model, r2 in result['model_r2s'].items():
                f.write(f"    {model}: R²={r2:.4f}\n")
            
            f.write("\n")
            
            # Command to run the sample
            cmd = f"python plot_prediction.py --architecture DenseNet169 --loss_fn {args.loss_fn} --optimizer {args.optimizer} --index {result['sample_idx']}"
            f.write(f"  Command: {cmd}\n\n")
    
    # Save as CSV
    csv_data = []
    for i, result in enumerate(detailed_results):
        row = {
            'Rank': i + 1,
            'Sample Index': result['sample_idx'],
            'Average R²': result['avg_r2'],
            'Minimum R²': result['min_r2'],
            'b value': result['b_value'],
            'Small Delta': result['small_delta'],
            'Big Delta': result['big_delta']
        }
        
        for model, r2 in result['model_r2s'].items():
            row[f'{model} R²'] = r2
            
        csv_data.append(row)
    
    pd.DataFrame(csv_data).to_csv(f"{output_dir}/top_common_samples_{args.loss_fn}_{args.optimizer}.csv", index=False)
    
    # Plot R² distribution
    plot_r2_distribution(results, list(models.keys()), output_dir)
    
    print(f"\nResults saved to {output_dir}")
    print("\nTop samples found:")
    for i, result in enumerate(detailed_results):
        print(f"Sample {i+1}: Index {result['sample_idx']}, Avg R²={result['avg_r2']:.4f}, MRI Params: b={result['b_value']}, δ={result['small_delta']}, Δ={result['big_delta']}")
    
    # Generate visualization commands for each sample
    print("\nTo visualize top samples (DenseNet169 example):")
    for i, result in enumerate(detailed_results):
        cmd = f"python plot_prediction.py --architecture DenseNet169 --loss_fn {args.loss_fn} --optimizer {args.optimizer} --index {result['sample_idx']}"
        print(cmd)
    
    # Also plot the top sample for each architecture
    print("\nCommands to plot the top sample for all architectures:")
    top_sample = detailed_results[0]['sample_idx']
    for architecture in models.keys():
        cmd = f"python plot_prediction.py --architecture {architecture} --loss_fn {args.loss_fn} --optimizer {args.optimizer} --index {top_sample}"
        print(cmd)

if __name__ == "__main__":
    main()