import os
import numpy as np
import pandas as pd
import torch
import argparse
import pickle
from monai.transforms import Compose, ToTensor
from torch.utils.data import Subset, DataLoader
import SimpleITK as sitk
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.dataset import TiffDataset, custom_collate

# Import models
from architectures.Regressor import CustomRegressor
from architectures.BasicUNet import BasicUNet
from architectures.AutoEncoder import AutoEncoder
from architectures.DenseNet import DenseNet169
from architectures.EfficientNet import EfficientNetBN

FONTSIZE = 20
TICK_FONTSIZE = 18
TITLE_FONTSIZE = 20
LEGEND_FONTSIZE = 18

def parse_args():
    parser = argparse.ArgumentParser(description="Generate error vs. MRI parameter plots")
    parser.add_argument('--architecture', type=str, default='Regressor',
                        help='Model architecture: Regressor, BasicUNet, AutoEncoder, DenseNet169, EfficientNetB0')
    parser.add_argument('--loss_fn', type=str, default='MSELoss',
                        help='Loss function used: L1Loss, MSELoss, CustomL1Loss')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='Optimizer used: Adam, AdamW')
    parser.add_argument('--output_dir', type=str, default='results/plots/error_vs_params',
                        help='Directory to save output figures')
    parser.add_argument('--force_evaluation', action='store_true',
                        help='Force re-evaluation even if results file exists')
    return parser.parse_args()

def load_model_and_data():
    # Load data_list and test indices
    with open(os.path.join('splits', 'data_list.pkl'), 'rb') as f:
        data_list = pickle.load(f)
    with open(os.path.join('splits', 'test_indices.pkl'), 'rb') as f:
        test_indices = pickle.load(f)

    test_transforms = Compose([ToTensor()])
    test_ds = TiffDataset(data_list, test_transforms)
    test_dataset = Subset(test_ds, test_indices)
    
    batch_size = 32
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=16, 
        collate_fn=custom_collate
    )
    
    return test_loader, len(test_dataset)

def create_model(architecture):
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
    elif architecture == "EfficientNetB0":
        model = EfficientNetBN(
            model_name="efficientnet-b0",
            spatial_dims=3,
            in_channels=9,
            num_classes=550,
            pretrained=False
        )
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    
    return model

def evaluate_model(model, test_loader, device):
    model.eval()
    
    # Initialize lists to store results
    mri_params_list = []
    mse_values = []
    mae_values = []
    r2_values = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating model"):
            images = batch['images'].to(device)
            mri_params = batch['mri_params'].to(device)
            labels = batch['label'].to(device)
            
            # Handle original_params which could be a list or tensor
            if isinstance(batch['original_params'], list):
                original_params = np.array(batch['original_params'])
            else:
                original_params = batch['original_params'].cpu().numpy()
            
            # Make predictions
            predictions = model(images, mri_params)
            
            # Calculate metrics for each sample in the batch
            for i in range(len(predictions)):
                pred = predictions[i].cpu().numpy()
                label = labels[i].cpu().numpy()
                
                # Calculate metrics
                mse = mean_squared_error(label.flatten(), pred.flatten())
                mae = mean_absolute_error(label.flatten(), pred.flatten())
                r2 = r2_score(label.flatten(), pred.flatten())
                
                # Store results
                mse_values.append(mse)
                mae_values.append(mae)
                r2_values.append(r2)
                mri_params_list.append(original_params[i])
    
    # Convert to numpy arrays
    mri_params_array = np.array(mri_params_list)
    
    # Create DataFrame with results
    results_df = pd.DataFrame({
        'b_value': mri_params_array[:, 0],
        'small_delta': mri_params_array[:, 1],
        'big_delta': mri_params_array[:, 2],
        'MSE': mse_values,
        'MAE': mae_values,
        'R2': r2_values
    })
    
    return results_df

def create_multi_metric_plots(results_df, architecture, loss_fn, optimizer, output_dir):
    # Create output directory if it doesn't exist
    output_dir = os.path.join(output_dir, f'{architecture}_{loss_fn}_{optimizer}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("paper", font_scale=1.5)
    
    # Define color palette
    palette = sns.color_palette("tab10")
    
    # 1. Combined subplot grid for b_value vs all metrics
    fig, axes = plt.subplots(1, 3, figsize=(22, 8), sharey=False)
    
    # MSE plot
    sns.scatterplot(x='b_value', y='MSE', data=results_df, s=80, alpha=0.7, color=palette[0], ax=axes[0])
    sns.regplot(x='b_value', y='MSE', data=results_df, scatter=False, color=palette[0], ax=axes[0])
    #axes[0].set_title('MSE vs. b-value')
    axes[0].set_xlabel('b-value')
    axes[0].set_ylabel('Mean Squared Error')
    
    # MAE plot
    sns.scatterplot(x='b_value', y='MAE', data=results_df, s=80, alpha=0.7, color=palette[1], ax=axes[1])
    sns.regplot(x='b_value', y='MAE', data=results_df, scatter=False, color=palette[1], ax=axes[1])
    #axes[1].set_title('MAE vs. b-value')
    axes[1].set_xlabel('b-value')
    axes[1].set_ylabel('Mean Absolute Error')
    
    # R2 plot
    sns.scatterplot(x='b_value', y='R2', data=results_df, s=80, alpha=0.7, color=palette[2], ax=axes[2])
    sns.regplot(x='b_value', y='R2', data=results_df, scatter=False, color=palette[2], ax=axes[2])
    #axes[2].set_title('R² vs. b-value')
    axes[2].set_xlabel('b-value')
    axes[2].set_ylabel('R² Score')
    
    plt.suptitle(f'Performance Metrics vs. b-value ({architecture}, {loss_fn}, {optimizer})', fontsize=16, y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{architecture}_{loss_fn}_{optimizer}_metrics_vs_b_value.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, f'{architecture}_{loss_fn}_{optimizer}_metrics_vs_b_value.pdf'))
    #plt.show()
    plt.close()
    
    # 2. Combined subplot grid for small_delta vs all metrics
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=False)
    
    # MSE plot
    sns.scatterplot(x='small_delta', y='MSE', data=results_df, s=80, alpha=0.7, color=palette[0], ax=axes[0])
    sns.regplot(x='small_delta', y='MSE', data=results_df, scatter=False, color=palette[0], ax=axes[0])
    #axes[0].set_title('MSE vs. δ')
    axes[0].set_xlabel('δ')
    axes[0].set_ylabel('Mean Squared Error')
    
    # MAE plot
    sns.scatterplot(x='small_delta', y='MAE', data=results_df, s=80, alpha=0.7, color=palette[1], ax=axes[1])
    sns.regplot(x='small_delta', y='MAE', data=results_df, scatter=False, color=palette[1], ax=axes[1])
    #axes[1].set_title('MAE vs. δ')
    axes[1].set_xlabel('δ')
    axes[1].set_ylabel('Mean Absolute Error')
    
    # R2 plot
    sns.scatterplot(x='small_delta', y='R2', data=results_df, s=80, alpha=0.7, color=palette[2], ax=axes[2])
    sns.regplot(x='small_delta', y='R2', data=results_df, scatter=False, color=palette[2], ax=axes[2])
    #axes[2].set_title('R² vs. δ')
    axes[2].set_xlabel('δ')
    axes[2].set_ylabel('R² Score')
    
    plt.suptitle(f'Performance Metrics vs. δ ({architecture}, {loss_fn}, {optimizer})', fontsize=16, y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{architecture}_{loss_fn}_{optimizer}_metrics_vs_small_delta.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, f'{architecture}_{loss_fn}_{optimizer}_metrics_vs_small_delta.pdf'))
    plt.close()
    
    # 3. Combined subplot grid for big_delta vs all metrics
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=False)
    
    # MSE plot
    sns.scatterplot(x='big_delta', y='MSE', data=results_df, s=80, alpha=0.7, color=palette[0], ax=axes[0])
    sns.regplot(x='big_delta', y='MSE', data=results_df, scatter=False, color=palette[0], ax=axes[0])
    #axes[0].set_title('MSE vs. Δ')
    axes[0].set_xlabel('Δ')
    axes[0].set_ylabel('Mean Squared Error')
    
    # MAE plot
    sns.scatterplot(x='big_delta', y='MAE', data=results_df, s=80, alpha=0.7, color=palette[1], ax=axes[1])
    sns.regplot(x='big_delta', y='MAE', data=results_df, scatter=False, color=palette[1], ax=axes[1])
    #axes[1].set_title('MAE vs. Δ')
    axes[1].set_xlabel('Δ')
    axes[1].set_ylabel('Mean Absolute Error')
    
    # R2 plot
    sns.scatterplot(x='big_delta', y='R2', data=results_df, s=80, alpha=0.7, color=palette[2], ax=axes[2])
    sns.regplot(x='big_delta', y='R2', data=results_df, scatter=False, color=palette[2], ax=axes[2])
    #axes[2].set_title('R² vs. Δ')
    axes[2].set_xlabel('Δ')
    axes[2].set_ylabel('R² Score')
    
    plt.suptitle(f'Performance Metrics vs. Δ ({architecture}, {loss_fn}, {optimizer})', fontsize=16, y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{architecture}_{loss_fn}_{optimizer}_metrics_vs_big_delta.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, f'{architecture}_{loss_fn}_{optimizer}_metrics_vs_big_delta.pdf'))
    plt.close()
    
    # 4. Create a comprehensive grid plot with all parameters and metrics
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))

    # Row 1: b_value
    sns.scatterplot(x='b_value', y='MSE', data=results_df, s=80, alpha=0.7, color=palette[0], ax=axes[0, 0])
    sns.regplot(x='b_value', y='MSE', data=results_df, scatter=False, color=palette[0], ax=axes[0, 0])
    #axes[0, 0].set_title('MSE vs. b-value')
    axes[0, 0].set_xlabel('b-value')
    axes[0, 0].set_ylabel('Mean Squared Error')

    sns.scatterplot(x='b_value', y='MAE', data=results_df, s=80, alpha=0.7, color=palette[1], ax=axes[0, 1])
    sns.regplot(x='b_value', y='MAE', data=results_df, scatter=False, color=palette[1], ax=axes[0, 1])
    #axes[0, 1].set_title('MAE vs. b-value')
    axes[0, 1].set_xlabel('b-value')
    axes[0, 1].set_ylabel('Mean Absolute Error')

    sns.scatterplot(x='b_value', y='R2', data=results_df, s=80, alpha=0.7, color=palette[2], ax=axes[0, 2])
    sns.regplot(x='b_value', y='R2', data=results_df, scatter=False, color=palette[2], ax=axes[0, 2])
    #axes[0, 2].set_title(r'$R^2$ vs. b-value')
    axes[0, 2].set_xlabel('b-value')
    axes[0, 2].set_ylabel(r'$R^2$ Score')

    # Row 2: small_delta
    sns.scatterplot(x='small_delta', y='MSE', data=results_df, s=80, alpha=0.7, color=palette[0], ax=axes[1, 0])
    sns.regplot(x='small_delta', y='MSE', data=results_df, scatter=False, color=palette[0], ax=axes[1, 0])
    #axes[1, 0].set_title(r'MSE vs. $\delta$')
    axes[1, 0].set_xlabel(r'$\delta$')
    axes[1, 0].set_ylabel('Mean Squared Error')

    sns.scatterplot(x='small_delta', y='MAE', data=results_df, s=80, alpha=0.7, color=palette[1], ax=axes[1, 1])
    sns.regplot(x='small_delta', y='MAE', data=results_df, scatter=False, color=palette[1], ax=axes[1, 1])
    #axes[1, 1].set_title(r'MAE vs. $\delta$')
    axes[1, 1].set_xlabel(r'$\delta$')
    axes[1, 1].set_ylabel('Mean Absolute Error')

    sns.scatterplot(x='small_delta', y='R2', data=results_df, s=80, alpha=0.7, color=palette[2], ax=axes[1, 2])
    sns.regplot(x='small_delta', y='R2', data=results_df, scatter=False, color=palette[2], ax=axes[1, 2])
    #axes[1, 2].set_title(r'$R^2$ vs. $\delta$')
    axes[1, 2].set_xlabel(r'$\delta$')
    axes[1, 2].set_ylabel(r'$R^2$ Score')

    # Row 3: big_delta
    sns.scatterplot(x='big_delta', y='MSE', data=results_df, s=80, alpha=0.7, color=palette[0], ax=axes[2, 0])
    sns.regplot(x='big_delta', y='MSE', data=results_df, scatter=False, color=palette[0], ax=axes[2, 0])
    #axes[2, 0].set_title(r'MSE vs. $\Delta$')
    axes[2, 0].set_xlabel(r'$\Delta$')
    axes[2, 0].set_ylabel('Mean Squared Error')

    sns.scatterplot(x='big_delta', y='MAE', data=results_df, s=80, alpha=0.7, color=palette[1], ax=axes[2, 1])
    sns.regplot(x='big_delta', y='MAE', data=results_df, scatter=False, color=palette[1], ax=axes[2, 1])
    #axes[2, 1].set_title(r'MAE vs. $\Delta$')
    axes[2, 1].set_xlabel(r'$\Delta$')
    axes[2, 1].set_ylabel('Mean Absolute Error')

    sns.scatterplot(x='big_delta', y='R2', data=results_df, s=80, alpha=0.7, color=palette[2], ax=axes[2, 2])
    sns.regplot(x='big_delta', y='R2', data=results_df, scatter=False, color=palette[2], ax=axes[2, 2])
    #axes[2, 2].set_title(r'$R^2$ vs. $\Delta$')
    axes[2, 2].set_xlabel(r'$\Delta$')
    axes[2, 2].set_ylabel(r'$R^2$ Score')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{architecture}_{loss_fn}_{optimizer}_comprehensive_analysis.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, f'{architecture}_{loss_fn}_{optimizer}_comprehensive_analysis.pdf'))
    plt.close()

    # Aggregate results by unique MRI parameter combinations before creating 3D plots
    aggregated_df = results_df.groupby(['b_value', 'small_delta', 'big_delta']).agg({
        'MSE': 'mean',
        'MAE': 'mean',
        'R2': 'mean'
    }).reset_index()
    
    print(f"\nAggregated to {len(aggregated_df)} unique parameter combinations (from {len(results_df)} samples)")

    # 5. Create 3D plots for each metric (using aggregated data)
    metrics = ['MSE', 'MAE', 'R2']
    fig = plt.figure(figsize=(20, 6))
    #list of color maps for the three metrics
    cmap = ['viridis', 'plasma', 'coolwarm']
    for i, metric in enumerate(metrics):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        
        scatter = ax.scatter(
            aggregated_df['b_value'],
            aggregated_df['small_delta'],
            aggregated_df['big_delta'],
            c=aggregated_df[metric],
            cmap=cmap[i],
            s=200,
            alpha=0.8,
            edgecolors='black',
            linewidths=0.5
        )
        
        ax.set_xlabel('b-value', labelpad=10, fontsize=FONTSIZE)
        ax.set_ylabel(r'$\delta$', labelpad=10, fontsize=FONTSIZE)
        ax.set_zlabel(r'$\Delta$', labelpad=10, fontsize=FONTSIZE)
        ax.tick_params(axis='x', labelsize=TICK_FONTSIZE)
        ax.tick_params(axis='y', labelsize=TICK_FONTSIZE)
        ax.tick_params(axis='z', labelsize=TICK_FONTSIZE)
        
        # Format tick labels to show one decimal place
        from matplotlib.ticker import FormatStrFormatter
        #ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        #ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        
        if metric == 'R2':
            ax.set_title(f'Average R$^2$', fontsize=TITLE_FONTSIZE)
        else:
            ax.set_title(f'Average {metric}', fontsize=TITLE_FONTSIZE)
        
        cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.12)
        cbar.ax.tick_params(labelsize=LEGEND_FONTSIZE)
        # Removed the line: cbar.set_label(metric)
    
    #plt.suptitle(f'Relationship between MRI parameters and model performance', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{architecture}_{loss_fn}_{optimizer}_3D_analysis.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, f'{architecture}_{loss_fn}_{optimizer}_3D_analysis.pdf'))
    plt.close()
    
    # 6. Create a correlation matrix
    plt.figure(figsize=(10, 8))
    corr_columns = ['b_value', 'small_delta', 'big_delta', 'MSE', 'MAE', 'R2']
    corr_matrix = results_df[corr_columns].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title(f'Correlation Matrix ({architecture}, {loss_fn}, {optimizer})')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{architecture}_{loss_fn}_{optimizer}_correlation_matrix.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, f'{architecture}_{loss_fn}_{optimizer}_correlation_matrix.pdf'))
    plt.close()
    
    # Save results to CSV
    results_df.to_csv(os.path.join(output_dir, f'{architecture}_{loss_fn}_{optimizer}_performance_data.csv'), index=False)

def main():
    args = parse_args()
    
    # Define results file path
    results_file = f"results/plots/error_vs_params/evaluation_results_{args.architecture}_{args.loss_fn}_{args.optimizer}.pkl"
    
    # Check if evaluation results already exist
    if os.path.exists(results_file) and not args.force_evaluation:
        print(f"Loading existing evaluation results from {results_file}")
        with open(results_file, 'rb') as f:
            results_df = pickle.load(f)
        print(f"Loaded results for {len(results_df)} test samples")
    else:
        print("Evaluation results not found. Running model evaluation...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Load test data
        test_loader, num_test_samples = load_model_and_data()
        print(f"Loaded {num_test_samples} test samples")
        
        # Create and load model
        model = create_model(args.architecture)
        model_path = f"models/best_{args.architecture}_{args.loss_fn}_{args.optimizer}.pth"
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        print(f"Loaded model from {model_path}")
        
        # Evaluate model
        results_df = evaluate_model(model, test_loader, device)
        print("Model evaluation complete")
        
        # Save evaluation results
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        with open(results_file, 'wb') as f:
            pickle.dump(results_df, f)
        print(f"Evaluation results saved to {results_file}")
    
    # Create and save plots
    create_multi_metric_plots(
        results_df, 
        args.architecture, 
        args.loss_fn, 
        args.optimizer, 
        args.output_dir
    )
    print(f"Plots saved to {args.output_dir}")

if __name__ == "__main__":
    main()