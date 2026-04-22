import os
import numpy as np
import pandas as pd
import pickle
import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

FONTSIZE = 20
TICK_FONTSIZE = 18
TITLE_FONTSIZE = 20
LEGEND_FONTSIZE = 18

def parse_args():
    parser = argparse.ArgumentParser(description="Generate comparative 3D error vs. MRI parameter plots")
    parser.add_argument('--output_dir', type=str, default='results/plots/error_vs_params',
                        help='Directory to save output figures')
    parser.add_argument('--model1_arch', type=str, default='DenseNet169',
                        help='First model architecture')
    parser.add_argument('--model1_loss', type=str, default='HuberLoss',
                        help='First model loss function')
    parser.add_argument('--model1_opt', type=str, default='Adam',
                        help='First model optimizer')
    parser.add_argument('--model2_arch', type=str, default='Regressor',
                        help='Second model architecture')
    parser.add_argument('--model2_loss', type=str, default='MSELoss',
                        help='Second model loss function')
    parser.add_argument('--model2_opt', type=str, default='Adam',
                        help='Second model optimizer')
    parser.add_argument('--model1_name', type=str, default='DenseNet169',
                        help='Display name for first model')
    parser.add_argument('--model2_name', type=str, default='ResNet',
                        help='Display name for second model')
    return parser.parse_args()

def load_results(results_file):
    """Load evaluation results from pickle file"""
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    with open(results_file, 'rb') as f:
        results_df = pickle.load(f)
    
    return results_df

def aggregate_results(results_df):
    """Aggregate results by unique MRI parameter combinations"""
    aggregated_df = results_df.groupby(['b_value', 'small_delta', 'big_delta']).agg({
        'MSE': 'mean',
        'MAE': 'mean',
        'R2': 'mean'
    }).reset_index()
    
    return aggregated_df

def create_comparison_3d_plots(df1, df2, model1_name, model2_name, output_dir):
    """Create side-by-side 3D comparison plots for two models"""
    
    # Aggregate both datasets
    agg_df1 = aggregate_results(df1)
    agg_df2 = aggregate_results(df2)
    
    print(f"\n{model1_name}: Aggregated to {len(agg_df1)} unique parameter combinations")
    print(f"{model2_name}: Aggregated to {len(agg_df2)} unique parameter combinations")
    
    metrics = ['MSE', 'MAE', 'R2']
    metric_labels = ['MSE', 'MAE', 'R²']
    cmaps = ['viridis', 'plasma', 'coolwarm']
    
    # Calculate global min/max for each metric to use the same scale
    vmin_values = {}
    vmax_values = {}
    
    for metric in metrics:
        vmin_values[metric] = min(agg_df1[metric].min(), agg_df2[metric].min())
        vmax_values[metric] = max(agg_df1[metric].max(), agg_df2[metric].max())
        print(f"{metric} range: [{vmin_values[metric]:.4f}, {vmax_values[metric]:.4f}]")
    
    # Create a figure with 2 rows (models) x 3 columns (metrics)
    fig = plt.figure(figsize=(24, 16))
    
    for i, metric in enumerate(metrics):
        # First row: Model 1
        ax1 = fig.add_subplot(2, 3, i+1, projection='3d')
        
        scatter1 = ax1.scatter(
            agg_df1['b_value'],
            agg_df1['small_delta'],
            agg_df1['big_delta'],
            c=agg_df1[metric],
            cmap=cmaps[i],
            s=200,
            alpha=0.8,
            edgecolors='black',
            linewidths=0.5,
            vmin=vmin_values[metric],
            vmax=vmax_values[metric]
        )
        
        ax1.set_xlabel('b-value', labelpad=10, fontsize=FONTSIZE)
        ax1.set_ylabel(r'$\delta$', labelpad=10, fontsize=FONTSIZE)
        ax1.set_zlabel(r'$\Delta$', labelpad=10, fontsize=FONTSIZE)
        ax1.tick_params(axis='x', labelsize=TICK_FONTSIZE)
        ax1.tick_params(axis='y', labelsize=TICK_FONTSIZE)
        ax1.tick_params(axis='z', labelsize=TICK_FONTSIZE)
        ax1.zaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        
        # Title: Show metric labels for all subplots
        ax1.set_title(f'{metric_labels[i]}', fontsize=TITLE_FONTSIZE, pad=20)
        
        cbar1 = plt.colorbar(scatter1, ax=ax1, fraction=0.046, pad=0.12)
        cbar1.ax.tick_params(labelsize=LEGEND_FONTSIZE)
        
        # Second row: Model 2
        ax2 = fig.add_subplot(2, 3, i+4, projection='3d')
        
        scatter2 = ax2.scatter(
            agg_df2['b_value'],
            agg_df2['small_delta'],
            agg_df2['big_delta'],
            c=agg_df2[metric],
            cmap=cmaps[i],
            s=200,
            alpha=0.8,
            edgecolors='black',
            linewidths=0.5,
            vmin=vmin_values[metric],
            vmax=vmax_values[metric]
        )
        
        ax2.set_xlabel('b-value', labelpad=10, fontsize=FONTSIZE)
        ax2.set_ylabel(r'$\delta$', labelpad=10, fontsize=FONTSIZE)
        ax2.set_zlabel(r'$\Delta$', labelpad=10, fontsize=FONTSIZE)
        ax2.tick_params(axis='x', labelsize=TICK_FONTSIZE)
        ax2.tick_params(axis='y', labelsize=TICK_FONTSIZE)
        ax2.tick_params(axis='z', labelsize=TICK_FONTSIZE)
        ax2.zaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        
        # No titles for second row
        
        cbar2 = plt.colorbar(scatter2, ax=ax2, fraction=0.046, pad=0.12)
        cbar2.ax.tick_params(labelsize=LEGEND_FONTSIZE)
    
    plt.tight_layout()
    
    # Add extra spacing at the top for model names
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.98, hspace=0.25, wspace=0.25)
    
    # Add model names as text annotations outside the plot area
    # Position them to the left of the first column
    fig.text(0.02, 0.75, model1_name, fontsize=TITLE_FONTSIZE+2, fontweight='bold', 
             rotation=90, verticalalignment='center', horizontalalignment='center')
    fig.text(0.02, 0.25, model2_name, fontsize=TITLE_FONTSIZE+2, fontweight='bold', 
             rotation=90, verticalalignment='center', horizontalalignment='center')
    
    # Save the comparison plot
    os.makedirs(output_dir, exist_ok=True)
    output_path_base = os.path.join(output_dir, f'comparison_{model1_name}_vs_{model2_name}_3D_analysis')
    plt.savefig(f'{output_path_base}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_path_base}.pdf', bbox_inches='tight')
    plt.savefig(f'{output_path_base}.svg', bbox_inches='tight')
    plt.close()
    
    print(f"\nComparison plots saved to {output_path_base}.[png/pdf/svg]")
    
    # Print statistics for comparison
    print("\n" + "="*60)
    print(f"COMPARISON STATISTICS")
    print("="*60)
    
    for metric in metrics:
        print(f"\n{metric}:")
        print(f"  {model1_name}: Mean={agg_df1[metric].mean():.6f}, Median={agg_df1[metric].median():.6f}, Std={agg_df1[metric].std():.6f}")
        print(f"  {model2_name}: Mean={agg_df2[metric].mean():.6f}, Median={agg_df2[metric].median():.6f}, Std={agg_df2[metric].std():.6f}")
        
        if metric == 'R2':
            diff = agg_df1[metric].mean() - agg_df2[metric].mean()
            print(f"  Difference: {diff:+.6f} ({'better' if diff > 0 else 'worse'} for {model1_name})")
        else:
            diff = agg_df1[metric].mean() - agg_df2[metric].mean()
            print(f"  Difference: {diff:+.6f} ({'better' if diff < 0 else 'worse'} for {model1_name})")

def main():
    args = parse_args()
    
    # Define results file paths
    results_file1 = f"results/plots/error_vs_params/evaluation_results_{args.model1_arch}_{args.model1_loss}_{args.model1_opt}.pkl"
    results_file2 = f"results/plots/error_vs_params/evaluation_results_{args.model2_arch}_{args.model2_loss}_{args.model2_opt}.pkl"
    
    print(f"Loading results for {args.model1_name}...")
    print(f"  File: {results_file1}")
    df1 = load_results(results_file1)
    print(f"  Loaded {len(df1)} samples")
    
    print(f"\nLoading results for {args.model2_name}...")
    print(f"  File: {results_file2}")
    df2 = load_results(results_file2)
    print(f"  Loaded {len(df2)} samples")
    
    # Create comparison plots
    create_comparison_3d_plots(
        df1, df2,
        args.model1_name,
        args.model2_name,
        args.output_dir
    )

if __name__ == "__main__":
    main()
