"""
Structure-Performance Analyzer for MRI Signal Prediction Models

This script identifies samples with varying prediction quality for a PRIMARY model,
then shows how ALL models perform on those same samples. This reveals how different
vascular structures affect prediction quality across architectures.

Main functionality:
- Evaluates ALL trained models on the test dataset
- For a PRIMARY model (e.g., DenseNet169), identifies samples across R² spectrum:
  * High R² samples (>= high_threshold, default 0.95) - easy structures
  * Medium R² samples (medium_low <= R² <= medium_high, default 0.70-0.85) - moderate structures
  * Low R² samples (< low_threshold, default 0.60) - challenging structures
- For each identified sample, shows performance of ALL other models
- Ranks samples within each category by the primary model's R²

Use case:
When you want to understand which vascular structures are challenging for a specific
model and whether those challenges are architecture-specific or universal.

Usage:
    python structure_performance_analyzer.py --primary_model DenseNet169 --optimizer Adam --loss_fn HuberLoss --samples_per_category 5

Arguments:
    --primary_model: The model to analyze (DenseNet169, Regressor, etc.)
    --optimizer: Optimizer used in training (Adam, AdamW)
    --loss_fn: Loss function used (L1Loss, MSELoss, HuberLoss, etc.)
    --samples_per_category: Number of samples to find in each R² category (default: 5)
    --high_threshold: Minimum R² for "high performance" category (default: 0.95)
    --medium_low: Minimum R² for "medium performance" category (default: 0.70)
    --medium_high: Maximum R² for "medium performance" category (default: 0.85)
    --low_threshold: Maximum R² for "low performance" category (default: 0.60)
    --dir: Data directory (default: 'data')

Output files in results/structure_analysis/:
    - structure_performance_{primary_model}_{loss_fn}_{optimizer}.txt (detailed report)
    - structure_performance_{primary_model}_{loss_fn}_{optimizer}.csv (tabular data)
    - r2_comparison_plot.png (comparison across models and categories)
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
import seaborn as sns
import sys

# Force UTF-8 stdout so non-ASCII characters in status prints don't crash
# under Windows' default cp1252 console encoding.
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
    parser = argparse.ArgumentParser(description="Analyze how vascular structure affects model performance")
    parser.add_argument('--primary_model', type=str, required=True,
                       help='Primary model to analyze: Regressor, DenseNet169, BasicUNet, AutoEncoder, EfficientNetB4')
    parser.add_argument('--optimizer', type=str, default='Adam', 
                       help='Optimizer used: Adam, AdamW')
    parser.add_argument('--loss_fn', type=str, default='HuberLoss',
                       help='Loss function used: L1Loss, MSELoss, CustomL1Loss, HuberLoss')
    parser.add_argument('--samples_per_category', type=int, default=5,
                       help='Number of samples to identify per R² category')
    parser.add_argument('--high_threshold', type=float, default=0.95,
                       help='Minimum R² for high performance category')
    parser.add_argument('--medium_low', type=float, default=0.70,
                       help='Minimum R² for medium performance category')
    parser.add_argument('--medium_high', type=float, default=0.85,
                       help='Maximum R² for medium performance category')
    parser.add_argument('--low_threshold', type=float, default=0.60,
                       help='Maximum R² for low performance category')
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

def evaluate_all_models(models, test_loader):
    """Evaluate all models and store R² for each sample"""
    results = []
    
    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating all models")):
        images = batch['images'].to(device)
        mri_params = batch['mri_params'].to(device)
        labels = batch['label']
        original_params = batch['original_params']
        animal_chunks = batch['animal_chunks']
        
        batch_size = images.shape[0]
        
        # For each sample in the batch
        for i in range(batch_size):
            sample_idx = batch_idx * test_loader.batch_size + i
            sample_result = {
                'sample_idx': sample_idx,
                'animal_chunk': animal_chunks[i],
                'original_params': original_params[i].numpy(),
                'b_value': original_params[i][0].item(),
                'small_delta': original_params[i][1].item(),
                'big_delta': original_params[i][2].item()
            }
            
            # Evaluate each model on this sample
            for model_name, model in models.items():
                with torch.no_grad():
                    prediction = model(images[i:i+1], mri_params[i:i+1]).cpu()
                
                r2 = r2_score(labels[i].numpy().flatten(), prediction.numpy().flatten())
                sample_result[f'{model_name}_r2'] = r2
            
            results.append(sample_result)
    
    return pd.DataFrame(results)

def categorize_samples(df, primary_model, args):
    """Categorize samples based on primary model's R²"""
    primary_col = f'{primary_model}_r2'
    
    # Define categories
    high_perf = df[df[primary_col] >= args.high_threshold].copy()
    medium_perf = df[(df[primary_col] >= args.medium_low) & 
                     (df[primary_col] <= args.medium_high)].copy()
    low_perf = df[df[primary_col] <= args.low_threshold].copy()
    
    # Sort by primary model's R² (descending for high/medium, ascending for low)
    high_perf = high_perf.sort_values(primary_col, ascending=False).head(args.samples_per_category)
    medium_perf = medium_perf.sort_values(primary_col, ascending=False).head(args.samples_per_category)
    low_perf = low_perf.sort_values(primary_col, ascending=True).head(args.samples_per_category)
    
    # Add category labels
    high_perf['category'] = 'High Performance'
    medium_perf['category'] = 'Medium Performance'
    low_perf['category'] = 'Low Performance'
    
    return pd.concat([high_perf, medium_perf, low_perf], ignore_index=True)

def plot_comparison(categorized_df, models, primary_model, output_dir):
    """Create visualization comparing model performance across categories"""
    model_cols = [f'{model}_r2' for model in models.keys()]
    
    # Prepare data for plotting
    plot_data = []
    for _, row in categorized_df.iterrows():
        for model in models.keys():
            plot_data.append({
                'Sample': f"Idx {int(row['sample_idx'])}",
                'Category': row['category'],
                'Model': model,
                'R² Score': row[f'{model}_r2'],
                'Primary Model': model == primary_model
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create figure with subplots for each category
    categories = ['High Performance', 'Medium Performance', 'Low Performance']
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    for idx, category in enumerate(categories):
        cat_data = plot_df[plot_df['Category'] == category]
        
        # Create grouped bar plot
        samples = cat_data['Sample'].unique()
        x = np.arange(len(samples))
        width = 0.8 / len(models)
        
        for i, model in enumerate(models.keys()):
            model_data = cat_data[cat_data['Model'] == model]
            r2_values = [model_data[model_data['Sample'] == s]['R² Score'].values[0] 
                        for s in samples]
            
            # Highlight primary model
            if model == primary_model:
                axes[idx].bar(x + i * width, r2_values, width, 
                            label=f'{model} (Primary)', alpha=0.9, edgecolor='black', linewidth=2)
            else:
                axes[idx].bar(x + i * width, r2_values, width, label=model, alpha=0.7)
        
        axes[idx].set_xlabel('Sample Index')
        axes[idx].set_ylabel('R² Score')
        axes[idx].set_title(f'{category}\n(Primary Model: {primary_model})')
        axes[idx].set_xticks(x + width * (len(models) - 1) / 2)
        axes[idx].set_xticklabels(samples, rotation=45, ha='right')
        axes[idx].legend()
        axes[idx].grid(alpha=0.3, axis='y')
        axes[idx].set_ylim([0, 1.0])
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/r2_comparison_plot.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    args = parse_args()
    
    # Load data_list and test indices
    with open(os.path.join('splits', 'data_list.pkl'), 'rb') as f:
        data_list = pickle.load(f)
    with open(os.path.join('splits', 'test_indices.pkl'), 'rb') as f:
        test_indices = pickle.load(f)
    
    # Define all architectures to evaluate
    all_architectures = ["Regressor", "DenseNet169", "BasicUNet", "AutoEncoder", "EfficientNetB4"]
    
    # Load models
    models = {}
    for architecture in all_architectures:
        model = get_model(architecture, args.optimizer, args.loss_fn)
        if model is not None:
            models[architecture] = model
            print(f"Loaded model: {architecture}")
        else:
            print(f"Model not found: {architecture}")
    
    if not models:
        print("No models found. Exiting.")
        return
    
    if args.primary_model not in models:
        print(f"Primary model {args.primary_model} not found. Available models: {list(models.keys())}")
        return
    
    # Create test dataset
    test_transforms = Compose([ToTensor()])
    test_ds = TiffDataset(data_list, transform=test_transforms, include_animal_chunk=True)
    test_dataset = Subset(test_ds, test_indices)
    
    # Create test dataloader
    batch_size = 8
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=custom_collate
    )
    
    # Evaluate all models on all samples
    print(f"Evaluating {len(models)} models on {len(test_dataset)} test samples...")
    results_df = evaluate_all_models(models, test_loader)
    
    # Categorize samples based on primary model performance
    print(f"\nCategorizing samples based on {args.primary_model} performance...")
    categorized_df = categorize_samples(results_df, args.primary_model, args)
    
    if categorized_df.empty:
        print("No samples found matching the criteria. Try adjusting thresholds.")
        return
    
    # Create output directory
    output_dir = "results/structure_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    output_file = f"{output_dir}/structure_performance_{args.primary_model}_{args.loss_fn}_{args.optimizer}.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Structure-Performance Analysis\n")
        f.write(f"Primary Model: {args.primary_model}\n")
        f.write(f"Loss Function: {args.loss_fn}\n")
        f.write(f"Optimizer: {args.optimizer}\n")
        f.write(f"Samples per category: {args.samples_per_category}\n\n")
        
        for category in ['High Performance', 'Medium Performance', 'Low Performance']:
            cat_samples = categorized_df[categorized_df['category'] == category]
            
            if cat_samples.empty:
                f.write(f"\n{category}: No samples found\n")
                continue
            
            f.write(f"\n{'='*80}\n")
            f.write(f"{category} ({len(cat_samples)} samples)\n")
            f.write(f"{'='*80}\n\n")
            
            for idx, row in cat_samples.iterrows():
                f.write(f"Sample {int(row['sample_idx'])}: {row['animal_chunk']}\n")
                f.write(f"  MRI Parameters: b={row['b_value']:.1f}, δ={row['small_delta']:.1f}, Δ={row['big_delta']:.1f}\n")
                f.write(f"  Primary Model ({args.primary_model}) R²: {row[f'{args.primary_model}_r2']:.4f}\n")
                f.write(f"  Other Models:\n")
                
                for model in models.keys():
                    if model != args.primary_model:
                        f.write(f"    {model}: R²={row[f'{model}_r2']:.4f}\n")
                
                f.write(f"\n  Visualization commands:\n")
                f.write(f"    # Using test index (may differ between scripts):\n")
                f.write(f"    python scripts/plot.py --architecture {args.primary_model} --loss_fn {args.loss_fn} --optimizer {args.optimizer} --index {int(row['sample_idx'])}\n")
                f.write(f"    \n")
                f.write(f"    # Verify it's the correct sample by checking:\n")
                f.write(f"    # - Animal/Chunk: {row['animal_chunk']}\n")
                f.write(f"    # - MRI params match: b={row['b_value']:.1f}, δ={row['small_delta']:.1f}, Δ={row['big_delta']:.1f}\n")
                f.write(f"    # - R² should be approximately {row[f'{args.primary_model}_r2']:.4f}\n\n")
    
    # Save CSV
    csv_file = f"{output_dir}/structure_performance_{args.primary_model}_{args.loss_fn}_{args.optimizer}.csv"
    categorized_df.to_csv(csv_file, index=False)
    
    # Create visualization
    plot_comparison(categorized_df, models, args.primary_model, output_dir)
    
    print(f"\n✓ Results saved to {output_dir}")
    print(f"✓ Text report: {output_file}")
    print(f"✓ CSV file: {csv_file}")
    print(f"✓ Visualization: {output_dir}/r2_comparison_plot.png")
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"Summary for {args.primary_model}:")
    print(f"{'='*80}")
    
    for category in ['High Performance', 'Medium Performance', 'Low Performance']:
        cat_samples = categorized_df[categorized_df['category'] == category]
        if not cat_samples.empty:
            print(f"\n{category}: {len(cat_samples)} samples")
            primary_col = f'{args.primary_model}_r2'
            print(f"  {args.primary_model} R² range: {cat_samples[primary_col].min():.4f} - {cat_samples[primary_col].max():.4f}")
            print(f"  Samples:")
            for _, row in cat_samples.iterrows():
                print(f"    - Index {int(row['sample_idx'])}: {row['animal_chunk']} (R²={row[primary_col]:.4f})")

if __name__ == "__main__":
    main()
