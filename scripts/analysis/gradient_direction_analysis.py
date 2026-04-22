import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Subset
import pickle
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tqdm import tqdm
import argparse
from monai.transforms import Compose, ToTensor

# Force UTF-8 stdout so non-ASCII characters don't crash under Windows cp1252.
try:
    sys.stdout.reconfigure(encoding='utf-8')
except (AttributeError, OSError):
    pass

# Make the project root importable so src/ and architectures/ resolve
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# For data loading
from src.dataset import TiffDataset, custom_collate

# Import model architectures
from architectures.Regressor import CustomRegressor
from architectures.BasicUNet import BasicUNet
from architectures.AutoEncoder import AutoEncoder
from architectures.DenseNet import DenseNet169
from architectures.EfficientNet import EfficientNetBN

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze prediction accuracy by gradient direction")
    parser.add_argument('--architecture', type=str, default='DenseNet169',
                        help='Model architecture: Regressor, BasicUNet, AutoEncoder, DenseNet169, EfficientNetB4')
    parser.add_argument('--loss_fn', type=str, default='HuberLoss',
                        help='Loss function: L1Loss, MSELoss, HuberLoss, CustomL1Loss')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='Optimizer: Adam, AdamW')
    parser.add_argument('--dir', type=str, default='data',
                        help='Directory of data')
    return parser.parse_args()

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

def analyze_gradient_directions(model, test_loader, device):
    # Define the phi-theta combinations for each gradient direction
    phi_theta_combinations = [
        (0, 0),     # Spin Echo - index 0
        (0, 90),    # index 1
        (45, 45),   # index 2
        (45, 90),   # index 3
        (45, 135),  # index 4
        (90, 45),   # index 5
        (90, 90),   # index 6
        (90, 135),  # index 7
        (135, 45),  # index 8
        (135, 90),  # index 9
        (135, 135)  # index 10
    ]
    
    # Initialize metrics for each direction
    direction_metrics = {idx: {"mse": [], "mae": [], "r2": []} for idx in range(len(phi_theta_combinations))}
    
    # Process each batch
    for batch in tqdm(test_loader, desc="Evaluating gradient directions"):
        images = batch['images'].to(device)
        mri_params = batch['mri_params'].to(device)
        labels = batch['label'].to(device)
        original_params = batch['original_params'].cpu().numpy()
        
        with torch.no_grad():
            predictions = model(images, mri_params)
        
        # Calculate metrics for each sample and direction
        for i in range(images.shape[0]):
            for direction_idx in range(len(phi_theta_combinations)):
                pred = predictions[i, direction_idx].cpu().numpy()
                truth = labels[i, direction_idx].cpu().numpy()
                
                # Calculate metrics
                mse = mean_squared_error(truth, pred)
                mae = mean_absolute_error(truth, pred)
                r2 = r2_score(truth, pred)
                
                # Store metrics
                direction_metrics[direction_idx]["mse"].append(mse)
                direction_metrics[direction_idx]["mae"].append(mae)
                direction_metrics[direction_idx]["r2"].append(r2)
    
    # Calculate aggregate statistics for each direction
    direction_summary = []
    for idx, metrics in direction_metrics.items():
        phi, theta = phi_theta_combinations[idx]
        direction_name = "Spin Echo" if idx == 0 else f"φ={phi}°, θ={theta}°"
        
        summary = {
            "Direction": direction_name,
            "Phi": phi,
            "Theta": theta,
            "MSE Mean": np.mean(metrics["mse"]),
            "MSE Std": np.std(metrics["mse"]),
            "MAE Mean": np.mean(metrics["mae"]),
            "MAE Std": np.std(metrics["mae"]),
            "R2 Mean": round(np.mean(metrics["r2"]), 2),  # Rounded to 2 decimal places
            "R2 Std": np.std(metrics["r2"]),
            "R2 >= 0.8": np.mean(np.array(metrics["r2"]) >= 0.8) * 100  # Percentage of samples with R² >= 0.8
        }
        direction_summary.append(summary)
    
    return pd.DataFrame(direction_summary), direction_metrics

def generate_latex_table(summary_df, args, output_dir):
    """Generate a LaTeX table from the summary dataframe"""
    
    # Format the table for LaTeX
    latex_table = "\\begin{table}[htbp]\n"
    latex_table += "\\centering\n"
    latex_table += "\\footnotesize\n"
    latex_table += f"\\caption{{Performance Metrics by Gradient Direction for {args.architecture} with {args.loss_fn}}}\n"
    latex_table += "\\begin{tabular}{lcccc}\n"
    latex_table += "\\toprule\n"
    latex_table += "\\textbf{Direction} & \\textbf{MSE $\\pm$ SD} & \\textbf{MAE $\\pm$ SD} & \\textbf{$R^{2}$ Score} & \\textbf{$R^{2} \\geq$ 0.8 (\\%)} \\\\\n"
    latex_table += "\\midrule\n"
    
    # Add each row
    for _, row in summary_df.iterrows():
        direction = row['Direction']
        mae = f"{row['MAE Mean']:.4f} $\\pm$ {row['MAE Std']:.4f}"
        mse = f"{row['MSE Mean']:.4f} $\\pm$ {row['MSE Std']:.4f}"
        r2 = f"{row['R2 Mean']:.2f}"  # 2 decimal places
        r2_threshold = f"{row['R2 >= 0.8']:.2f}"
        
        # Replace Greek symbols in direction with LaTeX commands
        direction = direction.replace("φ=", "$\\phi=$").replace("θ=", "$\\theta=$").replace("°", "$^{\\circ}$")
        
        # Format the direction cell differently for Spin Echo
        if "Spin Echo" in direction:
            direction = "\\textbf{" + direction + "}"
        
        # Add the row to the table
        latex_table += f"{direction} & {mse} & {mae} & {r2} & {r2_threshold} \\\\\n"
    
    latex_table += "\\bottomrule\n"
    latex_table += "\\end{tabular}\n"
    latex_table += "\\label{tab:direction_metrics}\n"
    latex_table += "\\end{table}"
    
    return latex_table

def plot_direction_metrics(summary_df, raw_metrics, args, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Bar plot of mean R² scores
    plt.figure(figsize=(12, 6))
    bars = plt.bar(summary_df['Direction'], summary_df['R2 Mean'], 
                  yerr=summary_df['R2 Std'], capsize=5, alpha=0.7)
    
    # Color the spin echo bar differently
    bars[0].set_color('crimson')
    for i in range(1, len(bars)):
        bars[i].set_color('steelblue')
    
    plt.axhline(y=0.8, color='r', linestyle='--', alpha=0.7, label='$R^{2} = 0.8$ threshold')
    plt.title(f'Mean $R^{{2}}$ Score by Gradient Direction\n{args.architecture} with {args.loss_fn}')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.ylabel('$R^{2}$ Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/direction_r2_scores_{args.architecture}_{args.loss_fn}_{args.optimizer}.png', dpi=300)
    plt.savefig(f'{output_dir}/direction_r2_scores_{args.architecture}_{args.loss_fn}_{args.optimizer}.svg', format='svg')
    plt.close()
    
    # 2. Percentage of samples with R² >= 0.8
    plt.figure(figsize=(12, 6))
    bars = plt.bar(summary_df['Direction'], summary_df['R2 >= 0.8'], alpha=0.7)
    bars[0].set_color('crimson')
    for i in range(1, len(bars)):
        bars[i].set_color('steelblue')
    
    plt.title(f'Percentage of Samples with $R^{{2}} \\geq 0.8$ by Gradient Direction\n{args.architecture} with {args.loss_fn}')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 100)
    plt.ylabel('Percentage (%)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/direction_r2_threshold_{args.architecture}_{args.loss_fn}_{args.optimizer}.png', dpi=300)
    plt.savefig(f'{output_dir}/direction_r2_threshold_{args.architecture}_{args.loss_fn}_{args.optimizer}.svg', format='svg')
    plt.close()
    
    # 3. Violin plots of R² distribution
    plt.figure(figsize=(14, 8))
    data_for_violin = []
    labels = []
    
    for idx, direction in enumerate(summary_df['Direction']):
        data_for_violin.append(raw_metrics[idx]['r2'])
        labels.append(direction)
    
    # Create a color list, with the first one (Spin Echo) different
    colors = ['crimson'] + ['steelblue'] * (len(data_for_violin) - 1)
    
    parts = plt.violinplot(data_for_violin, showmeans=False, showmedians=True)
    
    # Color the violins
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    
    # Set x-tick labels
    plt.xticks(np.arange(1, len(labels) + 1), labels, rotation=45, ha='right')
    
    # Add threshold line
    plt.axhline(y=0.8, color='r', linestyle='--', alpha=0.7, label='$R^{2} = 0.8$ threshold')
    
    plt.title(f'$R^{{2}}$ Score Distribution by Gradient Direction\n{args.architecture} with {args.loss_fn}')
    plt.ylabel('$R^{2}$ Score')
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/direction_r2_violin_{args.architecture}_{args.loss_fn}_{args.optimizer}.png', dpi=300)
    plt.savefig(f'{output_dir}/direction_r2_violin_{args.architecture}_{args.loss_fn}_{args.optimizer}.svg', format='svg')
    plt.close()
    
    # 4. Heatmap visualization of error distribution
    phi_values = [0, 45, 90, 135]
    theta_values = [0, 45, 90, 135]
    
    # Create a dictionary to map (phi, theta) to metrics
    phi_theta_map = {f"{row['Phi']}-{row['Theta']}": 
                      {"r2": row["R2 Mean"], "mse": row["MSE Mean"], "mae": row["MAE Mean"]} 
                      for _, row in summary_df.iterrows()}
    
    # Create matrices for heatmaps
    r2_matrix = np.zeros((len(phi_values), len(theta_values)))
    mse_matrix = np.zeros((len(phi_values), len(theta_values)))
    
    for i, phi in enumerate(phi_values):
        for j, theta in enumerate(theta_values):
            key = f"{phi}-{theta}"
            if key in phi_theta_map:
                r2_matrix[i, j] = phi_theta_map[key]["r2"]
                mse_matrix[i, j] = phi_theta_map[key]["mse"]
            else:
                r2_matrix[i, j] = np.nan
                mse_matrix[i, j] = np.nan
    
    # R² heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(r2_matrix, annot=True, fmt=".2f", cmap="YlGnBu", 
                xticklabels=theta_values, yticklabels=phi_values, vmin=0.7, vmax=1.0)
    plt.title(f'Mean $R^{{2}}$ Score by Gradient Angle\n{args.architecture} with {args.loss_fn}')
    plt.xlabel('$\\theta$ (degrees)')
    plt.ylabel('$\\phi$ (degrees)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/direction_r2_heatmap_{args.architecture}_{args.loss_fn}_{args.optimizer}.png', dpi=300)
    plt.savefig(f'{output_dir}/direction_r2_heatmap_{args.architecture}_{args.loss_fn}_{args.optimizer}.svg', format='svg')
    plt.close()
    
    # MSE heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(mse_matrix, annot=True, fmt=".4f", cmap="YlOrRd_r", 
                xticklabels=theta_values, yticklabels=phi_values)
    plt.title(f'Mean MSE by Gradient Angle\n{args.architecture} with {args.loss_fn}')
    plt.xlabel('$\\theta$ (degrees)')
    plt.ylabel('$\\phi$ (degrees)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/direction_mse_heatmap_{args.architecture}_{args.loss_fn}_{args.optimizer}.png', dpi=300)
    plt.savefig(f'{output_dir}/direction_mse_heatmap_{args.architecture}_{args.loss_fn}_{args.optimizer}.svg', format='svg')
    plt.close()

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
    
    # Set up test dataset and loader
    test_transforms = Compose([ToTensor()])
    test_ds = TiffDataset(data_list, transform=test_transforms)
    test_dataset = Subset(test_ds, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0, collate_fn=custom_collate, pin_memory=True)
    
    print(f"Loading model: {model_path}")
    # Load model
    model = load_model(args.architecture, model_path, device)
    print(f"Model loaded successfully. Starting evaluation on {len(test_dataset)} samples...")
    
    # Analyze gradient directions
    summary_df, raw_metrics = analyze_gradient_directions(model, test_loader, device)
    
    # Create output directory
    output_dir = f"results/direction_analysis/{args.architecture}_{args.loss_fn}_{args.optimizer}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the summary dataframe
    summary_df.to_csv(f'{output_dir}/direction_summary_{args.architecture}_{args.loss_fn}_{args.optimizer}.csv', index=False)
    
    # Generate and save LaTeX table
    latex_table = generate_latex_table(summary_df, args, output_dir)
    # Save the LaTeX table to a file with UTF-8 encoding
    with open(f'{output_dir}/direction_table_{args.architecture}_{args.loss_fn}_{args.optimizer}.tex', 'w', encoding='utf-8') as f:
        f.write(latex_table)
    
    # Plot the results
    plot_direction_metrics(summary_df, raw_metrics, args, output_dir)
    
    print(f"Analysis complete. Results saved to {output_dir}")
    print("\nGradient Direction Performance Summary:")
    print(summary_df.to_string(index=False))
    print("\nLaTeX Table generated and saved to the output directory.")
    
    print("\n" + "="*80)
    print("SUCCESS: Gradient direction analysis completed successfully!")
    print("="*80)

if __name__ == "__main__":
    main()