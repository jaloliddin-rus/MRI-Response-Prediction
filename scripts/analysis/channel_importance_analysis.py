import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Subset, Dataset
import pickle
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tqdm import tqdm
import argparse
from monai.transforms import Compose, ToTensor
import copy

# Make the project root importable so `src.*` and `architectures.*` resolve.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# For data loading
from src.dataset import TiffDataset, custom_collate

# Import model architectures
from architectures.Regressor import CustomRegressor
from architectures.DenseNet import DenseNet169
from architectures.BasicUNet import BasicUNet
from architectures.AutoEncoder import AutoEncoder
from architectures.EfficientNet import EfficientNetBN

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze input channel importance")
    parser.add_argument('--architecture', type=str, default='DenseNet169',
                        help='Model architecture: Regressor, BasicUNet, AutoEncoder, DenseNet169, EfficientNetB4')
    parser.add_argument('--loss_fn', type=str, default='HuberLoss',
                        help='Loss function: L1Loss, MSELoss, HuberLoss, CustomL1Loss')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='Optimizer: Adam, AdamW')
    parser.add_argument('--dir', type=str, default='data',
                        help='Directory of data')
    return parser.parse_args()

class ChannelAblationDataset(Dataset):
    """Dataset wrapper that zeros out specific channels"""
    def __init__(self, original_dataset, channel_to_ablate=None):
        self.original_dataset = original_dataset
        self.channel_to_ablate = channel_to_ablate
        
        # Channel names for reference
        self.channel_names = [
            "Binary Image", 
            "Magnetic Field Gradient",
            "Hematocrit (Hct)",
            "Oxygen Saturation (SatO2)",
            "Partial Oxygen Pressure (PO2)",
            "Velocity Magnitude",
            "Velocity X (vx)",
            "Velocity Y (vy)",
            "Velocity Z (vz)"
        ]
    
    def __len__(self):
        return len(self.original_dataset)
    
    def __getitem__(self, idx):
        item = self.original_dataset[idx]
        
        # Copy the images to avoid modifying the original
        images = item['images'].clone()
        
        # If channel_to_ablate is specified, zero out that channel
        if self.channel_to_ablate is not None:
            images[self.channel_to_ablate] = torch.zeros_like(images[self.channel_to_ablate])
        
        # Return the modified item
        return {
            "images": images,
            "mri_params": item['mri_params'],
            "label": item['label'],
            "original_params": item['original_params']
        }

def load_model(architecture, model_path, device):
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
        raise ValueError(f"Unknown architecture: {architecture}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def evaluate_model(model, test_loader, device):
    """Evaluate model performance on the test set"""
    all_predictions = []
    all_labels = []
    
    for batch in tqdm(test_loader, desc="Evaluating"):
        images = batch['images'].to(device)
        mri_params = batch['mri_params'].to(device)
        labels = batch['label'].to(device)
        
        with torch.no_grad():
            predictions = model(images, mri_params)
        
        all_predictions.append(predictions.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    
    # Concatenate all batches
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Calculate metrics
    r2 = r2_score(all_labels.flatten(), all_predictions.flatten())
    mse = mean_squared_error(all_labels.flatten(), all_predictions.flatten())
    mae = mean_absolute_error(all_labels.flatten(), all_predictions.flatten())
    
    return {
        "R2": r2,
        "MSE": mse,
        "MAE": mae
    }

def analyze_channel_importance(model, test_dataset, device, channel_names, batch_size=32):
    """Analyze the importance of each input channel"""
    results = []
    
    # Baseline - evaluate with all channels
    baseline_dataset = ChannelAblationDataset(test_dataset, channel_to_ablate=None)
    baseline_loader = DataLoader(baseline_dataset, batch_size=batch_size, shuffle=False, 
                                 num_workers=4, collate_fn=custom_collate, pin_memory=True)
    baseline_metrics = evaluate_model(model, baseline_loader, device)
    
    results.append({
        "Channel": "All Channels (Baseline)",
        "Channel Index": -1,
        "R2": baseline_metrics["R2"],
        "MSE": baseline_metrics["MSE"],
        "MAE": baseline_metrics["MAE"],
        "R2 Drop": 0.0,
        "MSE Increase": 0.0,
        "MAE Increase": 0.0
    })
    
    # Test each channel by ablating it
    for channel_idx in range(len(channel_names)):
        ablated_dataset = ChannelAblationDataset(test_dataset, channel_to_ablate=channel_idx)
        ablated_loader = DataLoader(ablated_dataset, batch_size=batch_size, shuffle=False, 
                                   num_workers=4, collate_fn=custom_collate, pin_memory=True)
        
        print(f"Testing without {channel_names[channel_idx]} (channel {channel_idx})...")
        ablated_metrics = evaluate_model(model, ablated_loader, device)
        
        # Calculate performance drop
        r2_drop = baseline_metrics["R2"] - ablated_metrics["R2"]
        mse_increase = ablated_metrics["MSE"] - baseline_metrics["MSE"]
        mae_increase = ablated_metrics["MAE"] - baseline_metrics["MAE"]
        
        results.append({
            "Channel": channel_names[channel_idx],
            "Channel Index": channel_idx,
            "R2": ablated_metrics["R2"],
            "MSE": ablated_metrics["MSE"],
            "MAE": ablated_metrics["MAE"],
            "R2 Drop": r2_drop,
            "MSE Increase": mse_increase,
            "MAE Increase": mae_increase
        })
    
    return pd.DataFrame(results)

def plot_channel_importance(results_df, args, output_dir):
    """Create visualizations of channel importance"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract baseline row
    baseline_row = results_df[results_df['Channel'] == "All Channels (Baseline)"]
    
    # Get non-baseline rows and sort by R2 drop
    non_baseline_df = results_df[results_df['Channel'] != "All Channels (Baseline)"]
    sorted_df = non_baseline_df.sort_values(by="R2 Drop", ascending=False)

    # Add this code for channel name shortening
    # Define mapping from long to short names
    channel_name_mapping = {
        "Binary Image": "Binary",
        "Magnetic Field Gradient": "Gradient",
        "Hematocrit (Hct)": "Hct",
        "Oxygen Saturation (SatO2)": "SatO$_2$",
        "Partial Oxygen Pressure (PO2)": "PO$_2$",
        "Velocity Magnitude": "Velocity",
        "Velocity X (vx)": "V$_x$",
        "Velocity Y (vy)": "V$_y$",
        "Velocity Z (vz)": "V$_z$"
    }
    
    # 1. Plot R2 Drop (higher drop = more important channel)
    plt.figure(figsize=(12, 8))
    bars = plt.bar(sorted_df['Channel'], sorted_df['R2 Drop'], color='steelblue', alpha=0.7)
    #plt.title(f'Channel Importance by $R^2$ Score Drop\n{args.architecture} with {args.loss_fn}', fontsize=14)
    plt.xlabel('Channel Removed', fontsize=12)
    plt.ylabel('$R^2$ Score Drop', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/channel_importance_r2_{args.architecture}_{args.loss_fn}_{args.optimizer}.png', dpi=300)
    plt.savefig(f'{output_dir}/channel_importance_r2_{args.architecture}_{args.loss_fn}_{args.optimizer}.svg', format='svg')
    plt.savefig(f'{output_dir}/channel_importance_r2_{args.architecture}_{args.loss_fn}_{args.optimizer}.pdf', format='pdf')
    plt.close()
    
    # 2. Plot MSE Increase (higher increase = more important channel)
    sorted_by_mse = non_baseline_df.sort_values(by="MSE Increase", ascending=False)
    plt.figure(figsize=(12, 8))
    bars = plt.bar(sorted_by_mse['Channel'], sorted_by_mse['MSE Increase'], color='indianred', alpha=0.7)
    #plt.title(f'Channel Importance by MSE Increase\n{args.architecture} with {args.loss_fn}', fontsize=14)
    plt.xlabel('Channel Removed', fontsize=12)
    plt.ylabel('MSE Increase', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/channel_importance_mse_{args.architecture}_{args.loss_fn}_{args.optimizer}.png', dpi=300)
    plt.savefig(f'{output_dir}/channel_importance_mse_{args.architecture}_{args.loss_fn}_{args.optimizer}.svg', format='svg')
    plt.savefig(f'{output_dir}/channel_importance_mse_{args.architecture}_{args.loss_fn}_{args.optimizer}.pdf', format='pdf')
    plt.close()
    
    # 3. Plot channel importance heatmap (normalized)
    # Normalize each metric column
    importance_metrics = ['R2 Drop', 'MSE Increase', 'MAE Increase']
    normalized_df = non_baseline_df.copy()
    
    for metric in importance_metrics:
        max_val = normalized_df[metric].max()
        if max_val > 0:  # Avoid division by zero
            normalized_df[f'{metric} (Normalized)'] = normalized_df[metric] / max_val
    
    # Create heatmap data
    channels = normalized_df['Channel'].tolist()
    shortened_channels = [channel_name_mapping.get(channel, channel) for channel in channels]
    metrics = [f'{m} (Normalized)' for m in importance_metrics]
    heatmap_data = normalized_df[metrics].values
    
    # 3. Plot channel importance heatmap (normalized)
    plt.figure(figsize=(6, 7))  # More square/vertical aspect ratio
    plt.rc('font', size=10)  # Slightly smaller font size

    # Create heatmap
    ax = sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        xticklabels=[m.split(' (')[0] for m in metrics],
        yticklabels=shortened_channels,
        square=True,  # Square cells
        linewidths=0.5,  # Thin lines between cells
        annot_kws={"size": 9},  # Smaller annotation font
        cbar_kws={"shrink": 0.8}  # Smaller colorbar
    )

    # Adjust layout to be tighter
    plt.tight_layout()
    plt.savefig(f'{output_dir}/channel_importance_heatmap_{args.architecture}_{args.loss_fn}_{args.optimizer}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/channel_importance_heatmap_{args.architecture}_{args.loss_fn}_{args.optimizer}.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(f'{output_dir}/channel_importance_heatmap_{args.architecture}_{args.loss_fn}_{args.optimizer}.svg', format='svg', bbox_inches='tight')
    plt.close()

def generate_latex_table(results_df, args, output_dir):
    """Generate a LaTeX table for channel importance results"""
    
    # Extract baseline row
    baseline_row = results_df[results_df['Channel'] == "All Channels (Baseline)"].iloc[0]
    
    # Get non-baseline rows and sort by R2 drop (most important first)
    non_baseline_df = results_df[results_df['Channel'] != "All Channels (Baseline)"]
    sorted_df = non_baseline_df.sort_values(by="R2 Drop", ascending=False)
    
    latex_table = "\\begin{table}[htbp]\n"
    latex_table += "\\centering\n"
    latex_table += "\\footnotesize\n"
    latex_table += f"\\caption{{Input Channel Importance Analysis for {args.architecture} with {args.loss_fn}}}\n"
    latex_table += "\\begin{tabular}{lccc}\n"
    latex_table += "\\toprule\n"
    latex_table += "\\textbf{Channel} & \\textbf{$R^2$ Drop} & \\textbf{MSE Increase} & \\textbf{MAE Increase} \\\\\n"
    latex_table += "\\midrule\n"
    
    # Add baseline row
    latex_table += f"\\textbf{{All Channels (Baseline)}} & {baseline_row['R2']:.4f} & {baseline_row['MSE']:.4f} & {baseline_row['MAE']:.4f} \\\\\n"
    latex_table += "\\midrule\n"
    
    # Add each row (sorted by importance)
    for _, row in sorted_df.iterrows():
        channel = row['Channel']
        r2_drop = row['R2 Drop']
        mse_increase = row['MSE Increase']
        mae_increase = row['MAE Increase']
        
        # Format with bold for significant importance
        if r2_drop > 0.05:  # Significant importance threshold
            channel = f"\\textbf{{{channel}}}"
        
        latex_table += f"{channel} & {r2_drop:.4f} & {mse_increase:.4f} & {mae_increase:.4f} \\\\\n"
    
    latex_table += "\\bottomrule\n"
    latex_table += "\\end{tabular}\n"
    latex_table += "\\label{tab:channel_importance}\n"
    latex_table += "\\end{table}"
    
    # Save the LaTeX table to a file with UTF-8 encoding
    with open(f'{output_dir}/channel_importance_table_{args.architecture}_{args.loss_fn}_{args.optimizer}.tex', 'w', encoding='utf-8') as f:
        f.write(latex_table)
    
    return latex_table

def main():
    args = parse_args()
    model_name = f"best_{args.architecture}_{args.loss_fn}_{args.optimizer}"
    model_path = f"models/{model_name}.pth"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data_list and test indices
    with open(os.path.join('splits', 'data_list.pkl'), 'rb') as f:
        data_list = pickle.load(f)
    with open(os.path.join('splits', 'test_indices.pkl'), 'rb') as f:
        test_indices = pickle.load(f)
    
    # Set up test dataset
    test_transforms = Compose([ToTensor()])
    test_ds = TiffDataset(data_list, transform=test_transforms)
    test_dataset = Subset(test_ds, test_indices)
    
    # Channel names for reference and plots
    channel_names = [
        "Binary Image", 
        "Magnetic Field Gradient",
        "Hematocrit (Hct)",
        "Oxygen Saturation (SatO2)",
        "Partial Oxygen Pressure (PO2)",
        "Velocity Magnitude",
        "Velocity X (vx)",
        "Velocity Y (vy)",
        "Velocity Z (vz)"
    ]
    
    # Load model
    print(f"Loading model: {model_path}")
    model = load_model(args.architecture, model_path, device)
    
    # Analyze channel importance
    print("Analyzing channel importance...")
    results_df = analyze_channel_importance(model, test_dataset, device, channel_names, batch_size=32)
    
    # Create output directory
    output_dir = f"results/channel_importance/{args.architecture}_{args.loss_fn}_{args.optimizer}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results to CSV
    results_df.to_csv(f'{output_dir}/channel_importance_{args.architecture}_{args.loss_fn}_{args.optimizer}.csv', index=False)
    
    # Generate plots
    plot_channel_importance(results_df, args, output_dir)
    
    # Generate and save LaTeX table
    latex_table = generate_latex_table(results_df, args, output_dir)
    
    print(f"Analysis complete. Results saved to {output_dir}")
    print("\nChannel Importance Ranking (by R2 Drop):")
    non_baseline_df = results_df[results_df['Channel'] != "All Channels (Baseline)"]
    sorted_by_importance = non_baseline_df.sort_values(by="R2 Drop", ascending=False)
    for i, (_, row) in enumerate(sorted_by_importance.iterrows()):
        print(f"{i+1}. {row['Channel']}: R2 Drop = {row['R2 Drop']:.4f}, MSE Increase = {row['MSE Increase']:.4f}")

if __name__ == "__main__":
    main()