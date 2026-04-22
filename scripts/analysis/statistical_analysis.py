#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Statistical Analysis Script for MRI Signal Prediction Models
Implements essential statistical tests for model comparison:
1. Friedman test + Nemenyi post-hoc analysis
2. Bootstrap confidence intervals (95% CI)

Author: Statistical Analysis Module
Created: January 2026
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import friedmanchisquare, wilcoxon
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except (AttributeError, OSError):
    pass

BASE_DIR = Path(__file__).resolve().parent.parent.parent


# Set plotting style
plt.rcParams["font.family"] = "Palatino Linotype"
plt.rcParams['font.size'] = 12
sns.set_style("whitegrid")


def load_model_results(results_dir=None):
    """
    Load MSE, MAE, and R² values for all models from pickle files
    
    Returns:
        dict: Dictionary with model names as keys and metrics as values
    """
    if results_dir is None:
        results_dir = os.path.join(BASE_DIR, 'results')
    
    print("=" * 80)
    print("Loading Model Results")
    print("=" * 80)
    
    mse_dir = os.path.join(results_dir, 'mse')
    mae_dir = os.path.join(results_dir, 'mae')
    metrics_dir = os.path.join(results_dir, 'evaluation', 'metrics')
    
    model_data = {}
    
    # Load MSE values
    print("\nLoading MSE values...")
    for filename in os.listdir(mse_dir):
        if filename.endswith('.pkl'):
            model_name = filename.replace('.pkl', '').replace('mse_', '')
            with open(os.path.join(mse_dir, filename), 'rb') as f:
                mse_values = pickle.load(f)
            
            if model_name not in model_data:
                model_data[model_name] = {}
            model_data[model_name]['MSE'] = np.array(mse_values)
            print(f"  ✓ {model_name}: {len(mse_values)} samples")
    
    # Load MAE values
    print("\nLoading MAE values...")
    for filename in os.listdir(mae_dir):
        if filename.endswith('.pkl'):
            model_name = filename.replace('.pkl', '').replace('mae_', '')
            with open(os.path.join(mae_dir, filename), 'rb') as f:
                mae_values = pickle.load(f)
            
            if model_name in model_data:
                model_data[model_name]['MAE'] = np.array(mae_values)
                print(f"  ✓ {model_name}: {len(mae_values)} samples")
    
    # Load R² values from metrics files
    print("\nLoading R² values from metrics files...")
    r2_data = {}
    for filename in os.listdir(metrics_dir):
        if filename.endswith('.txt'):
            model_name = filename.replace('.txt', '')
            with open(os.path.join(metrics_dir, filename), 'r') as f:
                content = f.read()
                # Extract overall R² score
                for line in content.split('\n'):
                    if 'Overall R2 Score:' in line:
                        r2_score = float(line.split(':')[1].strip())
                        r2_data[model_name] = r2_score
                        break
    
    # Since we don't have per-sample R² in pickle files, we'll compute them
    # from predictions if needed. For now, store the overall R² from metrics
    print("\nExtracted overall R² scores:")
    for model_name, r2 in r2_data.items():
        if model_name in model_data:
            model_data[model_name]['R2_overall'] = r2
            print(f"  ✓ {model_name}: R² = {r2:.4f}")
    
    print(f"\n✓ Successfully loaded data for {len(model_data)} models")
    return model_data


def friedman_test(model_data, metric='MSE'):
    """
    Perform Friedman test to compare multiple models
    
    Args:
        model_data: Dictionary containing model results
        metric: Metric to compare ('MSE', 'MAE')
    
    Returns:
        tuple: (statistic, p_value, model_names, data_matrix)
    """
    print(f"\n{'=' * 80}")
    print(f"Friedman Test for {metric}")
    print(f"{'=' * 80}")
    
    # Extract data for each model
    model_names = sorted(model_data.keys())
    data_matrix = []
    
    for model_name in model_names:
        if metric in model_data[model_name]:
            data_matrix.append(model_data[model_name][metric])
    
    # Ensure all models have the same number of samples
    min_samples = min(len(d) for d in data_matrix)
    data_matrix = [d[:min_samples] for d in data_matrix]
    
    # Perform Friedman test
    statistic, p_value = friedmanchisquare(*data_matrix)
    
    print(f"\nTest: Friedman chi-square test")
    print(f"Number of models: {len(model_names)}")
    print(f"Number of samples: {min_samples}")
    print(f"Statistic: {statistic:.4f}")
    print(f"p-value: {p_value:.6e}")
    
    if p_value < 0.05:
        print(f"✓ Result: SIGNIFICANT difference between models (p < 0.05)")
    else:
        print(f"✗ Result: No significant difference between models (p ≥ 0.05)")
    
    return statistic, p_value, model_names, data_matrix


def nemenyi_posthoc_test(model_data, metric='MSE', alpha=0.05):
    """
    Perform Nemenyi post-hoc test after Friedman test
    
    Args:
        model_data: Dictionary containing model results
        metric: Metric to compare
        alpha: Significance level
    
    Returns:
        DataFrame: Pairwise comparison results
    """
    print(f"\n{'=' * 80}")
    print(f"Nemenyi Post-Hoc Test for {metric}")
    print(f"{'=' * 80}")
    
    model_names = sorted(model_data.keys())
    data_matrix = []
    
    for model_name in model_names:
        if metric in model_data[model_name]:
            data_matrix.append(model_data[model_name][metric])
    
    min_samples = min(len(d) for d in data_matrix)
    data_matrix = [d[:min_samples] for d in data_matrix]
    data_array = np.array(data_matrix).T  # Transpose: samples × models
    
    # Compute average ranks for each model
    ranks = np.zeros((data_array.shape[0], data_array.shape[1]))
    for i in range(data_array.shape[0]):
        ranks[i] = stats.rankdata(data_array[i])
    
    avg_ranks = np.mean(ranks, axis=0)
    
    # Compute critical difference using Nemenyi test
    k = len(model_names)  # number of models
    n = min_samples  # number of samples
    
    # Critical value from studentized range statistic
    # For simplicity, using approximate critical value
    q_alpha = 2.728  # for k=5, alpha=0.05 (approximate)
    if k == 4:
        q_alpha = 2.639
    elif k == 3:
        q_alpha = 2.394
    
    cd = q_alpha * np.sqrt(k * (k + 1) / (6.0 * n))
    
    print(f"\nAverage ranks:")
    for name, rank in zip(model_names, avg_ranks):
        print(f"  {name}: {rank:.2f}")
    
    print(f"\nCritical Difference (CD): {cd:.4f}")
    
    # Pairwise comparisons
    results = []
    print(f"\nPairwise comparisons:")
    
    for i, j in combinations(range(k), 2):
        rank_diff = abs(avg_ranks[i] - avg_ranks[j])
        significant = rank_diff > cd
        
        results.append({
            'Model 1': model_names[i],
            'Model 2': model_names[j],
            'Rank Diff': rank_diff,
            'Critical Diff': cd,
            'Significant': 'Yes' if significant else 'No',
            'p-value': '<0.05' if significant else '≥0.05'
        })
        
        sig_marker = "*" if significant else " "
        print(f"  {model_names[i]} vs {model_names[j]}: "
              f"Δrank = {rank_diff:.4f} {sig_marker}")
    
    df_results = pd.DataFrame(results)
    return df_results, avg_ranks


def bootstrap_confidence_intervals(model_data, n_bootstrap=2000, confidence=0.95):
    """
    Compute bootstrap confidence intervals for MSE, MAE, and R² (if available)
    
    Args:
        model_data: Dictionary containing model results
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
    
    Returns:
        DataFrame: Confidence intervals for each model and metric
    """
    print(f"\n{'=' * 80}")
    print(f"Bootstrap Confidence Intervals ({int(confidence*100)}% CI)")
    print(f"{'=' * 80}")
    print(f"Bootstrap samples: {n_bootstrap}")
    
    results = []
    
    for model_name in sorted(model_data.keys()):
        print(f"\nProcessing {model_name}...")
        model_result = {'Model': model_name}
        
        for metric in ['MSE', 'MAE']:
            if metric in model_data[model_name]:
                data = model_data[model_name][metric]
                n = len(data)
                
                # Bootstrap resampling
                bootstrap_means = []
                for _ in range(n_bootstrap):
                    resample = np.random.choice(data, size=n, replace=True)
                    bootstrap_means.append(np.mean(resample))
                
                bootstrap_means = np.array(bootstrap_means)
                
                # Compute confidence intervals
                alpha = 1 - confidence
                lower_percentile = (alpha / 2) * 100
                upper_percentile = (1 - alpha / 2) * 100
                
                mean_val = np.mean(data)
                ci_lower = np.percentile(bootstrap_means, lower_percentile)
                ci_upper = np.percentile(bootstrap_means, upper_percentile)
                std_val = np.std(data)
                
                model_result[f'{metric}_mean'] = mean_val
                model_result[f'{metric}_std'] = std_val
                model_result[f'{metric}_CI_lower'] = ci_lower
                model_result[f'{metric}_CI_upper'] = ci_upper
                model_result[f'{metric}_CI_width'] = ci_upper - ci_lower
                
                print(f"  {metric}: {mean_val:.6f} "
                      f"[{ci_lower:.6f}, {ci_upper:.6f}]")
        
        # Add R² if available (overall score)
        if 'R2_overall' in model_data[model_name]:
            r2 = model_data[model_name]['R2_overall']
            model_result['R2_overall'] = r2
            print(f"  R² (overall): {r2:.4f}")
        
        results.append(model_result)
    
    df_results = pd.DataFrame(results)
    return df_results


def plot_nemenyi_heatmap(nemenyi_results, avg_ranks, model_names, model_data, 
                         metric='MSE', output_dir=None):
    """
    Create heatmap visualization of Nemenyi post-hoc test results
    Shows only best combination per architecture (by highest R²)
    """
    if output_dir is None:
        output_dir = os.path.join(BASE_DIR, 'results', 'figures')
    
    print(f"\n{'=' * 80}")
    print(f"Generating Nemenyi Heatmap for {metric} (Best Combinations)")
    print(f"{'=' * 80}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse model names and select best per architecture
    def parse_model_name(model_name):
        """Parse model name into architecture, loss, optimizer"""
        if model_name.startswith('best_'):
            model_name = model_name[5:]
        
        if '_' in model_name:
            parts = model_name.split('_')
            if len(parts) >= 3:
                arch = parts[0]
                loss = parts[1]
                optimizer = '_'.join(parts[2:])
            elif len(parts) == 2:
                arch = parts[0]
                loss = parts[1]
                optimizer = ''
            else:
                arch = model_name
                loss = ''
                optimizer = ''
        else:
            arch = model_name
            loss = ''
            optimizer = ''
        return arch, loss, optimizer
    
    # Group models by architecture
    architectures = {}
    for i, model_name in enumerate(model_names):
        arch, loss, opt = parse_model_name(model_name)
        
        if arch not in architectures:
            architectures[arch] = []
        
        r2 = model_data[model_name].get('R2_overall', -999)
        architectures[arch].append({
            'full_name': model_name,
            'arch': arch,
            'loss': loss,
            'optimizer': opt,
            'rank': avg_ranks[i],
            'r2': r2,
            'index': i
        })
    
    # Select best combination per architecture (highest R²)
    best_models = []
    best_indices = []
    display_labels = []
    
    print("\nBest combinations selected (by highest R²):")
    for arch in sorted(architectures.keys()):
        combos = architectures[arch]
        best = max(combos, key=lambda x: x['r2'])
        best_models.append(best['full_name'])
        best_indices.append(best['index'])
        
        # Create display label: Just the architecture name
        display_label = best['arch']
        # Rename Regressor to ResNet for publication
        if display_label == 'Regressor':
            display_label = 'ResNet'
        display_labels.append(display_label)
        
        print(f"  {best['arch']}: {best['loss']}/{best['optimizer']} "
              f"(R² = {best['r2']:.4f}, Rank = {best['rank']:.2f})")
    
    n_models = len(best_models)
    
    # Create rank difference matrix
    rank_diff_matrix = np.zeros((n_models, n_models))
    for i in range(n_models):
        for j in range(n_models):
            rank_diff_matrix[i, j] = abs(avg_ranks[best_indices[i]] - avg_ranks[best_indices[j]])
    
    # Get critical difference from nemenyi_results
    cd = nemenyi_results.iloc[0]['Critical Diff']
    
    # Create significance matrix (True where difference is significant)
    sig_matrix = rank_diff_matrix > cd
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create mask for diagonal
    mask = np.eye(n_models, dtype=bool)
    
    # Create annotations with * for significant differences
    annot_matrix = np.empty((n_models, n_models), dtype=object)
    for i in range(n_models):
        for j in range(n_models):
            if i == j:
                annot_matrix[i, j] = ''
            elif sig_matrix[i, j]:
                annot_matrix[i, j] = f'{rank_diff_matrix[i, j]:.3f}*'
            else:
                annot_matrix[i, j] = f'{rank_diff_matrix[i, j]:.3f}'
    
    # Plot heatmap
    sns.heatmap(rank_diff_matrix, 
                annot=annot_matrix, 
                fmt='',
                cmap='RdYlGn_r',
                center=cd,
                vmin=0,
                vmax=rank_diff_matrix.max(),
                xticklabels=display_labels,
                yticklabels=display_labels,
                cbar_kws={'label': 'Rank Difference'},
                linewidths=0.5,
                linecolor='gray',
                mask=mask,
                ax=ax,
                square=True)
    
    # Remove axis labels
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    
    plt.tight_layout()
    
    # Save the figure
    plot_name = f'nemenyi_heatmap_{metric.lower()}_best'
    plt.savefig(os.path.join(output_dir, f'{plot_name}.png'), 
               dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'{plot_name}.pdf'), 
               bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Nemenyi heatmap saved to {output_dir}/{plot_name}.png")
    print(f"  ℹ Heatmap shows rank differences between best configurations")
    print(f"  ℹ * indicates significant difference (Rank Diff. > {cd:.4f})")
    
    return best_models, display_labels


def generate_compact_bootstrap_table(bootstrap_results, output_dir=None):
    """
    Generate compact LaTeX table showing only best combination for each architecture
    Similar to the evaluation metrics table in the paper - matches gen_results.py logic
    Best model selected by HIGHEST R² (matching paper methodology)
    """
    if output_dir is None:
        output_dir = os.path.join(BASE_DIR, 'results', 'tables')
    
    print(f"\n{'=' * 80}")
    print("Generating Compact Bootstrap CI Table (Best Combinations)")
    print(f"{'=' * 80}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse model names to extract architecture, loss, and optimizer
    # Format: best_Architecture_LossFunction_Optimizer (e.g., "best_DenseNet169_HuberLoss_Adam")
    architectures = {}
    
    for _, row in bootstrap_results.iterrows():
        model_full = row['Model']
        
        # Remove 'best_' prefix if present
        if model_full.startswith('best_'):
            model_full = model_full[5:]  # Remove 'best_' prefix
        
        # Parse model name - expecting format: Arch_Loss_Optimizer
        if '_' in model_full:
            parts = model_full.split('_')
            if len(parts) >= 3:
                arch = parts[0]
                loss = parts[1]
                optimizer = '_'.join(parts[2:])  # Handle multi-word optimizers
            elif len(parts) == 2:
                arch = parts[0]
                loss = parts[1]
                optimizer = 'Unknown'
            else:
                arch = model_full
                loss = 'Unknown'
                optimizer = 'Unknown'
        else:
            arch = model_full
            loss = 'Unknown'
            optimizer = 'Unknown'
        
        if arch not in architectures:
            architectures[arch] = []
        
        architectures[arch].append({
            'full_name': row['Model'],  # Keep original name
            'arch': arch,
            'loss': loss,
            'optimizer': optimizer,
            'mse_mean': row.get('MSE_mean', float('inf')),
            'mse_ci_lower': row.get('MSE_CI_lower', 0),
            'mse_ci_upper': row.get('MSE_CI_upper', 0),
            'mae_mean': row.get('MAE_mean', float('inf')),
            'mae_ci_lower': row.get('MAE_CI_lower', 0),
            'mae_ci_upper': row.get('MAE_CI_upper', 0),
            'r2': row.get('R2_overall', -999)  # Default to very low if missing
        })
    
    # Select best combination for each architecture by HIGHEST R² (matching gen_results.py)
    best_combinations = []
    
    print("\nBest combinations selected (by highest R²):")
    for arch in sorted(architectures.keys()):
        combos = architectures[arch]
        # Sort by R² descending, take the best
        best = max(combos, key=lambda x: x['r2'])
        best_combinations.append((arch, best))
        print(f"  {arch}: {best['loss']} + {best['optimizer']} (R² = {best['r2']:.4f}, MSE = {best['mse_mean']:.6f})")
    
    # Sort alphabetically by architecture name
    best_combinations.sort(key=lambda x: x[0])
    
    # Generate compact LaTeX table
    latex_compact = """\\begin{table}[htbp]
\\centering
\\caption{Performance comparison of best model configurations per architecture with 95\\% bootstrap confidence intervals. Models selected by highest R\\textsuperscript{2} score for each architecture family}
\\label{tab:bootstrap_ci_best}
\\begin{tabularx}{\\textwidth}{Xcc}
\\hline
\\textbf{Architecture} & \\textbf{MSE (95\\% CI)} & \\textbf{MAE (95\\% CI)} \\\\
\\hline
"""
    
    for arch, best in best_combinations:
        arch_clean = best['arch'].replace('_', '\\_')
        
        mse_str = f"{best['mse_mean']:.4f}"
        mse_ci = f"[{best['mse_ci_lower']:.4f}, {best['mse_ci_upper']:.4f}]"
        
        mae_str = f"{best['mae_mean']:.4f}"
        mae_ci = f"[{best['mae_ci_lower']:.4f}, {best['mae_ci_upper']:.4f}]"
        
        # Format with line breaks for CI
        latex_compact += f"{arch_clean} & \\makecell{{{mse_str} \\\\ \\scriptsize {mse_ci}}} & \\makecell{{{mae_str} \\\\ \\scriptsize {mae_ci}}} \\\\\n"
    
    latex_compact += """\\hline
\\end{tabularx}
\\end{table}

% Note: This table requires \\usepackage{tabularx} and \\usepackage{makecell} in the preamble
% Alternatively, use this simpler version without makecell:

\\begin{table}[htbp]
\\centering
\\caption{Performance comparison of best model configurations per architecture with 95\\% bootstrap confidence intervals (simplified format). Models selected by highest R\\textsuperscript{2} score for each architecture family}
\\label{tab:bootstrap_ci_best_simple}
\\begin{tabularx}{\\textwidth}{Xccc}
\\hline
\\textbf{Architecture} & \\textbf{MSE} & \\textbf{95\\% CI} & \\textbf{MAE} \\\\
\\hline
"""
    
    for arch, best in best_combinations:
        arch_clean = best['arch'].replace('_', '\\_')
        
        mse_str = f"{best['mse_mean']:.4f}"
        mse_ci = f"[{best['mse_ci_lower']:.4f}, {best['mse_ci_upper']:.4f}]"
        mae_str = f"{best['mae_mean']:.4f}"
        
        latex_compact += f"{arch_clean} & {mse_str} & {mse_ci} & {mae_str} \\\\\n"
    
    latex_compact += """\\hline
\\end{tabularx}
\\end{table}

% Note: This table requires \\usepackage{tabularx} in the preamble
"""
    
    # Save the compact table
    with open(os.path.join(output_dir, 'bootstrap_ci_best.tex'), 'w') as f:
        f.write(latex_compact)
    
    print(f"  ✓ Compact bootstrap CI table saved to {output_dir}/bootstrap_ci_best.tex")
    print("  ℹ Two versions provided: one with makecell (stacked CI), one simple (inline CI)")
    print("  ℹ Selection method: HIGHEST R² per architecture (matches gen_results.py)")
    
    return best_combinations


def generate_latex_tables(nemenyi_results, bootstrap_results, output_dir=None):
    """
    Generate LaTeX tables for publication (essential tests only)
    """
    if output_dir is None:
        output_dir = os.path.join(BASE_DIR, 'results', 'tables')
    
    print(f"\n{'=' * 80}")
    print("Generating LaTeX Tables")
    print(f"{'=' * 80}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Bootstrap CI table
    latex_bootstrap = """\\begin{table}[htbp]
\\centering
\\caption{Bootstrap confidence intervals for all model configurations showing mean squared error (MSE) and mean absolute error (MAE) with 95\\% confidence intervals computed from 2000 bootstrap resamples}
\\label{tab:bootstrap_ci}
\\begin{tabularx}{\\textwidth}{Xcccc}
\\hline
\\textbf{Model} & \\textbf{MSE} & \\textbf{95\\% CI (MSE)} & \\textbf{MAE} & \\textbf{95\\% CI (MAE)} \\\\
\\hline
"""
    
    for _, row in bootstrap_results.iterrows():
        model = row['Model'].replace('_', '\\_')
        if 'MSE_mean' in row:
            mse = f"{row['MSE_mean']:.6f}"
            mse_ci = f"[{row['MSE_CI_lower']:.6f}, {row['MSE_CI_upper']:.6f}]"
            mae = f"{row['MAE_mean']:.6f}"
            mae_ci = f"[{row['MAE_CI_lower']:.6f}, {row['MAE_CI_upper']:.6f}]"
            latex_bootstrap += f"{model} & {mse} & {mse_ci} & {mae} & {mae_ci} \\\\\n"
    
    latex_bootstrap += """\\hline
\\end{tabularx}
\\end{table}

% Note: This table requires \\usepackage{tabularx} in the preamble
"""
    
    with open(os.path.join(output_dir, 'bootstrap_ci.tex'), 'w') as f:
        f.write(latex_bootstrap)
    print("  ✓ Bootstrap CI table saved")
    
    print(f"\n✓ LaTeX tables saved to {output_dir}/")


def main():
    """
    Main function to run all statistical analyses
    """
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS FOR MRI SIGNAL PREDICTION MODELS")
    print("=" * 80)
    
    # Create output directories
    os.makedirs(os.path.join(BASE_DIR, 'results', 'tables'), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'results', 'figures'), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'results', 'statistical_tests'), exist_ok=True)
    
    # Load model results
    model_data = load_model_results()
    
    if not model_data:
        print("\n❌ Error: No model results found!")
        print("Make sure you have run validation.py for all models first.")
        return
    
    # Store all results
    all_results = {}
    
    # 1. Friedman Test for MSE
    print("\n" + "=" * 80)
    print("PART 1: FRIEDMAN TEST")
    print("=" * 80)
    
    friedman_stat_mse, friedman_p_mse, model_names, _ = friedman_test(model_data, 'MSE')
    friedman_stat_mae, friedman_p_mae, _, _ = friedman_test(model_data, 'MAE')
    
    all_results['friedman'] = {
        'MSE': {'statistic': friedman_stat_mse, 'p_value': friedman_p_mse},
        'MAE': {'statistic': friedman_stat_mae, 'p_value': friedman_p_mae}
    }
    
    # 2. Nemenyi Post-Hoc Test
    print("\n" + "=" * 80)
    print("PART 2: NEMENYI POST-HOC TEST")
    print("=" * 80)
    
    nemenyi_results_mse, avg_ranks_mse = nemenyi_posthoc_test(model_data, 'MSE')
    nemenyi_results_mae, avg_ranks_mae = nemenyi_posthoc_test(model_data, 'MAE')
    
    all_results['nemenyi'] = {
        'MSE': nemenyi_results_mse,
        'MAE': nemenyi_results_mae
    }
    
    # 3. Bootstrap Confidence Intervals
    print("\n" + "=" * 80)
    print("PART 3: BOOTSTRAP CONFIDENCE INTERVALS")
    print("=" * 80)
    
    bootstrap_results = bootstrap_confidence_intervals(model_data, n_bootstrap=2000)
    all_results['bootstrap'] = bootstrap_results
    
    # Find best model based on average rank from Nemenyi test
    best_model_idx = np.argmin(avg_ranks_mse)
    best_model = model_names[best_model_idx]
    
    print(f"\n{'=' * 80}")
    print(f"BEST MODEL IDENTIFIED: {best_model}")
    print(f"Average Rank: {avg_ranks_mse[best_model_idx]:.2f}")
    print(f"{'=' * 80}")
    
    # Save all results to CSV
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    nemenyi_results_mse.to_csv(os.path.join(BASE_DIR, 'results', 'statistical_tests', 'nemenyi_mse.csv'), index=False)
    nemenyi_results_mae.to_csv(os.path.join(BASE_DIR, 'results', 'statistical_tests', 'nemenyi_mae.csv'), index=False)
    bootstrap_results.to_csv(os.path.join(BASE_DIR, 'results', 'statistical_tests', 'bootstrap_ci.csv'), index=False)
    
    print("\n✓ CSV files saved to results/statistical_tests/")
    
    # Generate Nemenyi heatmaps (best combinations only)
    print("\n" + "=" * 80)
    print("PART 4: NEMENYI HEATMAP VISUALIZATION")
    print("=" * 80)
    
    best_models_mse, labels_mse = plot_nemenyi_heatmap(
        nemenyi_results_mse, avg_ranks_mse, model_names, model_data, 
        metric='MSE', output_dir=os.path.join(BASE_DIR, 'results', 'figures')
    )
    
    best_models_mae, labels_mae = plot_nemenyi_heatmap(
        nemenyi_results_mae, avg_ranks_mae, model_names, model_data,
        metric='MAE', output_dir=os.path.join(BASE_DIR, 'results', 'figures')
    )
    
    # Generate LaTeX tables
    generate_latex_tables(nemenyi_results_mse, bootstrap_results)
    
    # Generate compact bootstrap table (best combinations only)
    best_combinations = generate_compact_bootstrap_table(bootstrap_results)
    
    # Save summary report
    with open(os.path.join(BASE_DIR, 'results', 'statistical_tests', 'summary_report.txt'), 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("STATISTICAL ANALYSIS SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. FRIEDMAN TEST\n")
        f.write("-" * 80 + "\n")
        f.write(f"MSE: chi^2 = {friedman_stat_mse:.4f}, p = {friedman_p_mse:.6e}\n")
        f.write(f"MAE: chi^2 = {friedman_stat_mae:.4f}, p = {friedman_p_mae:.6e}\n")
        f.write(f"Conclusion: {'Significant' if friedman_p_mse < 0.05 else 'Not significant'} "
                f"differences between models\n\n")
        
        f.write("2. NEMENYI POST-HOC TEST (MSE)\n")
        f.write("-" * 80 + "\n")
        f.write("Average Ranks:\n")
        for name, rank in zip(model_names, avg_ranks_mse):
            f.write(f"  {name}: {rank:.2f}\n")
        f.write(f"\nBest model: {best_model} (rank: {avg_ranks_mse[best_model_idx]:.2f})\n\n")
        
        f.write("3. BOOTSTRAP CONFIDENCE INTERVALS\n")
        f.write("-" * 80 + "\n")
        f.write(bootstrap_results.to_string() + "\n\n")
        
        f.write("4. BEST MODEL SELECTION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Best performing model: {best_model}\n")
        f.write(f"Average rank: {avg_ranks_mse[best_model_idx]:.2f}\n")
        f.write(f"R² score: {model_data[best_model]['R2_overall']:.4f}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("Analysis complete!\n")
        f.write("Statistical tests performed: Friedman + Nemenyi + Bootstrap CI\n")
        f.write("=" * 80 + "\n")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  ✓ results/statistical_tests/nemenyi_mse.csv")
    print("  ✓ results/statistical_tests/nemenyi_mae.csv")
    print("  ✓ results/statistical_tests/bootstrap_ci.csv")
    print("  ✓ results/tables/bootstrap_ci.tex (full table)")
    print("  ✓ results/tables/bootstrap_ci_best.tex (compact - best combinations)")
    print("  ✓ results/figures/nemenyi_heatmap_mse_best.png/pdf")
    print("  ✓ results/figures/nemenyi_heatmap_mae_best.png/pdf")
    print("  ✓ results/statistical_tests/summary_report.txt")
    print("\nStatistical tests performed:")
    print("  1. Friedman test (global comparison)")
    print("  2. Nemenyi post-hoc test (pairwise comparisons)")
    print("  3. Bootstrap confidence intervals (uncertainty estimation)")
    print("  4. Nemenyi heatmap visualization (best combinations)")
    print(f"\nBest model: {best_model} (rank: {avg_ranks_mse[best_model_idx]:.2f})")
    print("\nYou can now include these results in your paper!")
    

if __name__ == "__main__":
    main()
