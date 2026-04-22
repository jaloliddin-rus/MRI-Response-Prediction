import os
import pickle
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

FONTSIZE = 16
TICK_FONTSIZE = 14
TITLE_FONTSIZE = 16
LEGEND_FONTSIZE = 14

def parse_args():
    parser = argparse.ArgumentParser(description="Generate boxenplots for best models")
    parser.add_argument('--mse_dir', default="results/mse", help='Directory containing MSE pickle files')
    parser.add_argument('--mae_dir', default="results/mae", help='Directory containing MAE pickle files')
    parser.add_argument('--output_dir', default="results/plots/boxplots", help='Directory to save plots')
    parser.add_argument('--best_models_file', default="results/tables/evaluation_metrics_best_full.csv", 
                        help='CSV file with best model configurations')
    return parser.parse_args()

def load_best_models(file_path):
    """Load the best model configurations from CSV file"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Best models file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    best_models = []
    
    for _, row in df.iterrows():
        model_info = {
            'architecture': row['Architecture'],
            'loss_function': row['Loss Function'],
            'optimizer': row['Optimizer']
        }
        best_models.append(model_info)
    
    return best_models

def get_metric_file_path(metric_dir, model_info, metric_prefix):
    """Construct the file path for a model's metric pickle file"""
    # Format: mse_best_Architecture_LossFunction_Optimizer.pkl
    file_name = f"{metric_prefix}_best_{model_info['architecture']}_{model_info['loss_function']}_{model_info['optimizer']}.pkl"
    return os.path.join(metric_dir, file_name)

def load_metric_data(file_path):
    """Load metric data from pickle file"""
    if not os.path.exists(file_path):
        print(f"Warning: Metric file not found: {file_path}")
        return None
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    return data

def create_boxenplot(data_dict, metric_name, output_path_base):
    """Create and save a boxenplot for the given metric data"""
    plt.figure(figsize=(12, 8))
    
    # Prepare data for plotting
    plot_data = []
    labels = []
    
    for model_name, values in data_dict.items():
        if values is not None:
            # Convert to numpy array if not already
            values_array = np.array(values)
            # Explicitly handle inf values
            values_filtered = values_array[~np.isinf(values_array)]
            # Filter out NaN values
            values_filtered = values_filtered[~np.isnan(values_filtered)]
            
            if len(values_filtered) > 0:
                plot_data.append(values_filtered)
                # Just use the architecture name for the label
                arch_name = model_name.split('_')[0]
                labels.append(arch_name)
    
    # Create the boxenplot
    if plot_data:
        ax = sns.boxenplot(data=plot_data, orient='h')
        
        # Customize the plot
        plt.yticks(range(len(labels)), labels)
        plt.xlabel(f"{metric_name} Value")
        plt.title(f"{metric_name} Distribution for Best Model Configurations")
        plt.grid(True, axis='x', linestyle='--', alpha=0.7)
        
        # Add median values as text annotations
        for i, values in enumerate(plot_data):
            median = np.median(values)
            ax.text(median, i, f"{median:.4f}", verticalalignment='center', 
                    fontsize=FONTSIZE, fontweight='bold', color='black', 
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path_base), exist_ok=True)
        
        # Save the figure in both PDF and SVG formats
        plt.tight_layout()
        plt.savefig(f"{output_path_base}.pdf", bbox_inches='tight')
        plt.savefig(f"{output_path_base}.svg", bbox_inches='tight')
        plt.close()
        
        print(f"Boxenplot saved to {output_path_base}.pdf and {output_path_base}.svg")
    else:
        print(f"No valid data for {metric_name} boxenplot")

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load best model configurations
    best_models = load_best_models(args.best_models_file)
    
    # Dictionaries to store metric data for each model
    mse_data = {}
    mae_data = {}
    
    # Load MSE and MAE data for each best model
    for model_info in best_models:
        model_name = f"{model_info['architecture']}_{model_info['loss_function']}_{model_info['optimizer']}"
        
        # Load MSE data
        mse_file = get_metric_file_path(args.mse_dir, model_info, "mse")
        mse_values = load_metric_data(mse_file)
        mse_data[model_name] = mse_values
        
        # Load MAE data
        mae_file = get_metric_file_path(args.mae_dir, model_info, "mae")
        mae_values = load_metric_data(mae_file)
        mae_data[model_name] = mae_values
    
    # Create boxenplots
    #mse_plot_path = os.path.join(args.output_dir, "mse_boxenplot")
    #create_boxenplot(mse_data, "MSE", mse_plot_path)
    
    #mae_plot_path = os.path.join(args.output_dir, "mae_boxenplot")
    #create_boxenplot(mae_data, "MAE", mae_plot_path)
    
    # Optional: Create a combined plot with both metrics
    plt.figure(figsize=(16, 8))  # Wider but shorter for better text visibility
    
    # Create subplot for MSE with adjusted margins
    plt.subplot(1, 2, 1)
    mse_plot_data = []
    mse_labels = []
    
    for model_name, values in mse_data.items():
        if values is not None:
            values_array = np.array(values)
            # Explicitly handle inf values
            values_filtered = values_array[~np.isinf(values_array)]
            # Filter out NaN values
            values_filtered = values_filtered[~np.isnan(values_filtered)]
            
            if len(values_filtered) > 0:
                mse_plot_data.append(values_filtered)
                arch_name = model_name.split('_')[0]
                # Rename Regressor to ResNet
                if arch_name == 'Regressor':
                    arch_name = 'ResNet'
                mse_labels.append(arch_name)
    
    if mse_plot_data:
        ax1 = sns.boxenplot(data=mse_plot_data, orient='h')
        plt.yticks(range(len(mse_labels)), mse_labels, fontsize=18)
        plt.xlabel("MSE", fontsize=20)
        plt.xticks(fontsize=18)
        #plt.title("MSE Distribution", fontsize=FONTSIZE, pad=15)
        plt.grid(True, axis='x', linestyle='--', alpha=0.7)
        
        # Reduce margins for this subplot
        ax1.margins(y=0.02)
        
        # Add median values with improved visibility (median is standard for boxplots)
        for i, values in enumerate(mse_plot_data):
            median = np.median(values)
            if i == 0:
                ax1.text(median, i, f"Median: {median:.4f}", verticalalignment='center', 
                    horizontalalignment='center', fontsize=11, fontweight='bold', color='black',
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='gray', linewidth=0.5, boxstyle='round,pad=0.3'))
            else:
                ax1.text(median, i, f"{median:.4f}", verticalalignment='center', 
                    horizontalalignment='center', fontsize=11, fontweight='bold', color='black',
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='gray', linewidth=0.5, boxstyle='round,pad=0.3'))
    
    # Create subplot for MAE
    plt.subplot(1, 2, 2)
    mae_plot_data = []
    mae_labels = []
    
    for model_name, values in mae_data.items():
        if values is not None:
            values_array = np.array(values)
            # Explicitly handle inf values
            values_filtered = values_array[~np.isinf(values_array)]
            # Filter out NaN values
            values_filtered = values_filtered[~np.isnan(values_filtered)]
            
            if len(values_filtered) > 0:
                mae_plot_data.append(values_filtered)
                arch_name = model_name.split('_')[0]
                # Rename Regressor to ResNet
                if arch_name == 'Regressor':
                    arch_name = 'ResNet'
                mae_labels.append(arch_name)
    
    if mae_plot_data:
        ax2 = sns.boxenplot(data=mae_plot_data, orient='h')
        
        # Remove y-tick labels on the right subplot but keep the ticks visible
        plt.yticks(range(len(mae_labels)), [""] * len(mae_labels))
        
        plt.xlabel("MAE", fontsize=20)
        plt.xticks(fontsize=18)
        #plt.title("MAE Distribution", fontsize=FONTSIZE, pad=15)
        plt.grid(True, axis='x', linestyle='--', alpha=0.7)
        
        # Reduce margins for this subplot
        ax2.margins(y=0.02)
        
        # Add median values with improved visibility (median is standard for boxplots)
        for i, values in enumerate(mae_plot_data):
            median = np.median(values)
            if i == 0:
                ax2.text(median, i, f"Median: {median:.4f}", verticalalignment='center', 
                    horizontalalignment='center', fontsize=11, fontweight='bold', color='black',
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='gray', linewidth=0.5, boxstyle='round,pad=0.3'))
            else:
                ax2.text(median, i, f"{median:.4f}", verticalalignment='center', 
                    horizontalalignment='center', fontsize=11, fontweight='bold', color='black',
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='gray', linewidth=0.5, boxstyle='round,pad=0.3'))
    
    # Save combined plot with optimized spacing
    plt.tight_layout(pad=1.0, w_pad=0.2, h_pad=1.0)
    
    # Further adjust subplot spacing manually for better text visibility - minimal wspace for tighter layout
    plt.subplots_adjust(left=0.10, right=0.98, top=0.9, bottom=0.15, wspace=0.05)
    
    combined_plot_path = os.path.join(args.output_dir, "combined_MSE_MAE_metrics_boxenplot")
    plt.savefig(f"{combined_plot_path}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{combined_plot_path}.svg", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    if os.path.exists(f"{combined_plot_path}.pdf") and os.path.exists(f"{combined_plot_path}.svg"):
        print(f"Combined plot saved to {combined_plot_path}.pdf and {combined_plot_path}.svg")

if __name__ == "__main__":
    main()