#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep Learning Benchmarking Script
Compatible with Python 3.10 and PyTorch environment

Run this after the signal generation benchmarking script to compare times.

Author: Analysis Script
Created: 2024
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import pickle
import time
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from monai.transforms import Compose, ToTensor
from torch.utils.data import Subset, DataLoader
from statistics import mean, stdev
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import Dict, Optional, Tuple
# Import architectures and shared dataset / collate
from architectures.Regressor import CustomRegressor
from architectures.BasicUNet import BasicUNet
from architectures.AutoEncoder import AutoEncoder
from architectures.DenseNet import DenseNet169
from architectures.EfficientNet import EfficientNetBN
from src.dataset import TiffDataset, custom_collate


def _parse_float(value) -> Optional[float]:
    """Best-effort conversion to float for values that may be '0.92', 'N/A', etc."""
    try:
        if value is None:
            return None
        if isinstance(value, (int, float, np.floating)):
            return float(value)
        s = str(value).strip()
        if s.lower() in {"n/a", "na", "none", "--", ""}:
            return None
        return float(s)
    except Exception:
        return None


def load_precomputed_evaluation_metrics(
    metrics_csv_path: str = os.path.join('results', 'tables', 'evaluation_metrics_full.csv')
) -> Optional[pd.DataFrame]:
    """Load precomputed evaluation metrics (R² etc.) for all model/loss/optimizer combinations."""
    if not os.path.exists(metrics_csv_path):
        return None

    df = pd.read_csv(metrics_csv_path)
    # Normalize expected columns
    # Required: Architecture, Loss Function, Optimizer, R²
    expected = {'Architecture', 'Loss Function', 'Optimizer', 'R²'}
    if not expected.issubset(set(df.columns)):
        return None

    df = df.copy()
    df['R2_numeric'] = df['R²'].apply(_parse_float)
    return df


def get_best_combinations_from_metrics(metrics_df: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    """Pick best (loss, optimizer) per architecture using the highest precomputed R²."""
    best: Dict[str, Dict[str, object]] = {}
    for arch, group in metrics_df.groupby('Architecture'):
        group_valid = group.dropna(subset=['R2_numeric']).sort_values('R2_numeric', ascending=False)
        if group_valid.empty:
            continue
        row = group_valid.iloc[0]
        best[str(arch)] = {
            'loss': str(row['Loss Function']),
            'optimizer': str(row['Optimizer']),
            'r2': float(row['R2_numeric'])
        }
    return best


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark deep learning prediction times and compare with signal generation.")
    parser.add_argument('--signal_gen_results', type=str, default='timing_results/signal_generation_times.pkl',
                        help='Path to signal generation results pickle file')
    parser.add_argument('--dl_batch_size', type=int, default=32,
                        help='Batch size for deep learning prediction')
    parser.add_argument('--output_dir', type=str, default='timing_results',
                        help='Directory to save comparison results')
    return parser.parse_args()

def benchmark_deep_learning_prediction(batch_size):
    """Benchmark deep learning model prediction time for best performing combinations"""
    print("=== Deep Learning Prediction Benchmarking ===")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Load best combinations from precomputed evaluation metrics (so figures match your reported results)
    metrics_df = load_precomputed_evaluation_metrics()
    if metrics_df is None:
        print("⚠ Warning: Could not load precomputed metrics from results/tables/evaluation_metrics_full.csv")
        print("  Falling back to hard-coded R² values (may not match your paper / reported results).")
        best_combinations = {
            'AutoEncoder': {'loss': 'MSELoss', 'optimizer': 'AdamW', 'r2': 0.87},
            'BasicUNet': {'loss': 'MSELoss', 'optimizer': 'Adam', 'r2': 0.89},
            'DenseNet169': {'loss': 'HuberLoss', 'optimizer': 'Adam', 'r2': 0.92},
            'EfficientNetB4': {'loss': 'CustomL1Loss', 'optimizer': 'Adam', 'r2': 0.88},
            'Regressor': {'loss': 'CustomL1Loss', 'optimizer': 'Adam', 'r2': 0.88}
        }
    else:
        best_combinations = get_best_combinations_from_metrics(metrics_df)
        print("✓ Loaded precomputed evaluation metrics")
        print("  Using best (loss, optimizer) per architecture based on highest precomputed R²")
    
    # Load test data
    try:
        with open(os.path.join('splits', 'data_list.pkl'), 'rb') as f:
            data_list = pickle.load(f)
        with open(os.path.join('splits', 'test_indices.pkl'), 'rb') as f:
            test_indices = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Error: Required data files not found: {e}")
        print("Make sure splits/data_list.pkl and splits/test_indices.pkl exist")
        return None
    
    test_transforms = Compose([ToTensor()])
    test_ds = TiffDataset(data_list, transform=test_transforms)
    test_dataset = Subset(test_ds, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=0, pin_memory=True, collate_fn=custom_collate)

    print(f"Total benchmark samples (expanded test set): {len(test_dataset)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    dl_results = {}
    
    for arch, combo_info in best_combinations.items():
        print(f"\nBenchmarking {arch} with {combo_info['loss']} + {combo_info['optimizer']} (R² = {combo_info['r2']})...")
        
        # Load model
        try:
            if arch == "Regressor":
                model = CustomRegressor()
            elif arch == "BasicUNet":
                model = BasicUNet(spatial_dims=3, in_channels=9, out_channels=11)
            elif arch == "AutoEncoder":
                model = AutoEncoder(spatial_dims=3, in_channels=9, out_channels=32,
                                  channels=(32, 64, 128, 256), strides=(2, 2, 2, 2))
            elif arch == "DenseNet169":
                model = DenseNet169(spatial_dims=3, in_channels=9, out_channels=11,
                                  init_features=64, growth_rate=32, dropout_prob=0.2)
            elif arch == "EfficientNetB4":
                model = EfficientNetBN(model_name="efficientnet-b4", spatial_dims=3,
                                     in_channels=9, num_classes=550, pretrained=False)
        except Exception as e:
            print(f"❌ Error loading {arch} model: {e}")
            continue
        
        # Try to load trained weights for the best combination
        model_path = f"models/best_{arch}_{combo_info['loss']}_{combo_info['optimizer']}.pth"
        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location=device))
                print(f"  ✓ Loaded trained weights from {model_path}")
            except Exception as e:
                print(f"  ⚠ Error loading weights: {e}, using random weights")
        else:
            print(f"  ⚠ Warning: No trained weights found at {model_path}, using random weights")
        
        model.to(device)
        model.eval()
        
        prediction_times = []
        total_samples = 0
        
        # Warmup
        print("  - Warming up GPU...")
        with torch.no_grad():
            for _ in range(3):
                dummy_batch = next(iter(test_loader))
                dummy_images = dummy_batch['images'][:1].to(device)
                dummy_mri_params = dummy_batch['mri_params'][:1].to(device)
                _ = model(dummy_images, dummy_mri_params)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
        
        print("  - Benchmarking prediction times...")
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Testing {arch}"):
                images = batch['images'].to(device)
                mri_params = batch['mri_params'].to(device)
                
                # Measure time
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                batch_start_time = time.time()
                predictions = model(images, mri_params)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                batch_end_time = time.time()
                
                batch_time = batch_end_time - batch_start_time
                batch_size_actual = images.shape[0]
                total_samples += batch_size_actual
                
                # Time per sample
                time_per_sample = batch_time / batch_size_actual
                prediction_times.extend([time_per_sample] * batch_size_actual)
        
        if prediction_times:
            dl_results[arch] = {
                'times': prediction_times,
                'total_samples': total_samples,
                'avg_time': mean(prediction_times),
                'std_time': stdev(prediction_times) if len(prediction_times) > 1 else 0,
                'loss_function': combo_info['loss'],
                'optimizer': combo_info['optimizer'],
                'r2_score': combo_info['r2']
            }
            
            avg_time = mean(prediction_times)
            std_time = stdev(prediction_times) if len(prediction_times) > 1 else 0
            print(f"  ✓ Average prediction time: {avg_time:.6f} ± {std_time:.6f} seconds per sample")
        else:
            print(f"  ❌ No prediction times recorded for {arch}")
    
    return dl_results

def create_comparison_plots(signal_gen_results, dl_results, output_dir):
    """Create comparison plots and tables for best performing combinations"""
    
    # Extract signal generation statistics
    signal_stats = signal_gen_results['statistics']
    avg_signal_gen_time = signal_stats['mean']
    std_signal_gen_time = signal_stats['std']
    min_signal_gen_time = signal_stats['min']
    max_signal_gen_time = signal_stats['max']
    
    # Create comparison DataFrame
    comparison_data = []
    
    # Add signal generation data
    comparison_data.append({
        'Method': 'Physics-based Simulation',
        'Configuration': 'VirtualMRI + Graph Processing',
        'Average Time (s)': avg_signal_gen_time,
        'Std Time (s)': std_signal_gen_time,
        'Min Time (s)': min_signal_gen_time,
        'Max Time (s)': max_signal_gen_time,
        'R² Score': 'N/A',
        'Speedup': 1.0  # Reference
    })
    
    # Add deep learning results for best combinations
    for arch, results in dl_results.items():
        speedup = avg_signal_gen_time / results['avg_time']
        comparison_data.append({
            'Method': f'{arch}',
            'Configuration': f"{results['loss_function']} + {results['optimizer']}",
            'Average Time (s)': results['avg_time'],
            'Std Time (s)': results['std_time'],
            'Min Time (s)': min(results['times']),
            'Max Time (s)': max(results['times']),
            'R² Score': f"{results['r2_score']:.2f}",
            'Speedup': speedup
        })
    
    # Sort by R² score (DL models only) for better presentation
    df = pd.DataFrame(comparison_data)
    
    # Separate physics-based and DL results for sorting
    physics_row = df[df['Method'] == 'Physics-based Simulation']
    dl_rows = df[df['Method'] != 'Physics-based Simulation'].copy()
    dl_rows['R² Score Numeric'] = dl_rows['R² Score'].astype(float)
    dl_rows = dl_rows.sort_values('R² Score Numeric', ascending=False).drop('R² Score Numeric', axis=1)
    
    # Recombine
    df = pd.concat([physics_row, dl_rows]).reset_index(drop=True)
    
    # Save CSV
    df.to_csv(os.path.join(output_dir, 'timing_comparison_best_combinations.csv'), index=False)
    
    # Create bar plot for average times with better formatting
    plt.figure(figsize=(14, 8))
    methods = [f"{row['Method']}\n({row['Configuration']})" if 'Simulation' not in row['Method'] 
              else row['Method'] for _, row in df.iterrows()]
    avg_times = df['Average Time (s)'].tolist()
    std_times = df['Std Time (s)'].tolist()
    
    bars = plt.bar(range(len(methods)), avg_times, yerr=std_times, capsize=5, alpha=0.8)
    plt.xlabel('Method')
    plt.ylabel('Average Time per Sample (s)')
    plt.yscale('log')  # Use log scale due to large differences
    plt.xticks(range(len(methods)), methods, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Color the bars - red for physics, different shades of blue for DL models
    bars[0].set_color('crimson')  # Physics-based simulation
    colors = ['steelblue', 'royalblue', 'mediumblue', 'darkblue', 'midnightblue']
    for i in range(1, len(bars)):
        bars[i].set_color(colors[(i-1) % len(colors)])
    
    # Add R² scores as text on DL bars
    for i, (bar, (_, row)) in enumerate(zip(bars[1:], df.iloc[1:].iterrows()), 1):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 2,
                f"R² = {row['R² Score']}", ha='center', va='bottom', 
                fontsize=9, fontweight='bold', rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'timing_comparison_best_combinations.png'), 
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'timing_comparison_best_combinations.pdf'), 
                bbox_inches='tight')
    plt.close()
    
    # Create speedup plot for DL models only
    plt.figure(figsize=(12, 8))
    dl_methods = [row['Method'] for _, row in df.iloc[1:].iterrows()]
    speedups = df['Speedup'].tolist()[1:]
    r2_scores = [float(row['R² Score']) for _, row in df.iloc[1:].iterrows()]
    
    # Create bars with colors based on R² performance
    bars = plt.bar(range(len(dl_methods)), speedups, alpha=0.8)
    
    # Color bars based on R² performance (darker = better)
    max_r2 = max(r2_scores)
    min_r2 = min(r2_scores)
    for bar, r2 in zip(bars, r2_scores):
        # Normalize R² to color intensity (0.4 to 1.0)
        intensity = 0.4 + 0.6 * ((r2 - min_r2) / (max_r2 - min_r2))
        bar.set_color(plt.cm.Blues(intensity))
    
    plt.xlabel('Deep Learning Architecture')
    plt.ylabel('Speedup Factor (× faster than physics-based)')
    plt.xticks(range(len(dl_methods)), dl_methods, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add speedup values and R² scores on bars
    for i, (bar, speedup, r2) in enumerate(zip(bars, speedups, r2_scores)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(speedups)*0.01,
                f'{speedup:.0f}×\n(R² = {r2:.2f})', ha='center', va='bottom', 
                fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'speedup_comparison_best_combinations.png'), 
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'speedup_comparison_best_combinations.pdf'), 
                bbox_inches='tight')
    plt.close()
    
    return df

def generate_latex_table(df):
    """Generate LaTeX table for the timing comparison of best combinations"""
    latex_str = """\\begin{table}[htbp]
\\centering
\\footnotesize
\\caption{Computational time comparison: physics-based simulation vs deep learning prediction (best performing combinations)}
\\label{tab:timing_comparison_best}
\\begin{tabular}{llccccc}
\\hline
\\textbf{Method} & \\textbf{Configuration} & \\textbf{Avg Time (s)} & \\textbf{Std (s)} & \\textbf{R$^2$ Score} & \\textbf{Speedup} \\\\
\\hline
"""
    
    for _, row in df.iterrows():
        method = row['Method'].replace('_', '\\_')
        config = row['Configuration'].replace('_', '\\_')
        
        # Format timing values based on magnitude
        avg_time = f"{row['Average Time (s)']:.6f}" if row['Average Time (s)'] < 0.001 else f"{row['Average Time (s)']:.4f}"
        std_time = f"{row['Std Time (s)']:.6f}" if row['Std Time (s)'] < 0.001 else f"{row['Std Time (s)']:.4f}"
        
        r2_score = row['R² Score'] if row['R² Score'] != 'N/A' else '--'
        speedup = f"{row['Speedup']:.0f}\\times" if row['Speedup'] != 1.0 else "1\\times"
        
        latex_str += f"{method} & {config} & {avg_time} & {std_time} & {r2_score} & {speedup} \\\\\n"
    
    latex_str += """\\hline
\\multicolumn{6}{l}{\\footnotesize Note: Deep learning models show best performing loss function and optimizer combinations.} \\\\
\\multicolumn{6}{l}{\\footnotesize Speedup calculated as: (Physics-based time) / (Deep learning time).} \\\\
\\end{tabular}
\\end{table}
"""
    
    return latex_str

def load_signal_generation_results(signal_gen_results_path):
    """Load signal generation results from pickle file"""
    try:
        with open(signal_gen_results_path, 'rb') as f:
            results = pickle.load(f)
        
        print("✓ Signal generation results loaded successfully")
        print(f"  - Samples processed: {results['statistics']['count']}")
        print(f"  - Average time: {results['statistics']['mean']:.2f} ± {results['statistics']['std']:.2f} seconds")
        print(f"  - Total time: {results['statistics']['total']:.2f} seconds")
        
        return results
    except FileNotFoundError:
        print(f"❌ Error: Signal generation results file not found: {signal_gen_results_path}")
        print("Please run the signal generation benchmarking script first in your Python 3.7 environment")
        return None
    except Exception as e:
        print(f"❌ Error loading signal generation results: {e}")
        return None

def main():
    args = parse_args()
    
    print("=== Deep Learning vs Physics-based Signal Generation Comparison ===")
    print(f"Signal generation results: {args.signal_gen_results}")
    print(f"DL batch size: {args.dl_batch_size}")
    print(f"Output directory: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load signal generation results
    signal_gen_results = load_signal_generation_results(args.signal_gen_results)
    if signal_gen_results is None:
        return 1
    
    # Benchmark deep learning prediction
    print("\nPhase 2: Benchmarking deep learning predictions for best combinations...")
    dl_results = benchmark_deep_learning_prediction(args.dl_batch_size)
    
    if not dl_results:
        print("❌ No deep learning results obtained!")
        return 1
    
    # Print results summary
    signal_stats = signal_gen_results['statistics']
    print(f"\n=== RESULTS SUMMARY (Best Performing Combinations) ===")
    print(f"Physics-based simulation: {signal_stats['mean']:.4f} ± {signal_stats['std']:.4f} seconds")
    
    for arch, results in dl_results.items():
        speedup = signal_stats['mean'] / results['avg_time']
        print(f"{arch} ({results['loss_function']} + {results['optimizer']}, R² = {results['r2_score']:.2f}): "
              f"{results['avg_time']:.6f} ± {results['std_time']:.6f} seconds (Speedup: {speedup:.0f}×)")
    
    # Create comparison plots and save results
    print(f"\nSaving results to {args.output_dir}...")
    df = create_comparison_plots(signal_gen_results, dl_results, args.output_dir)
    
    # Generate LaTeX table
    latex_table = generate_latex_table(df)
    with open(os.path.join(args.output_dir, 'timing_comparison_best_combinations_table.tex'), 'w') as f:
        f.write(latex_table)
    
    # Save combined results
    combined_results = {
        'signal_generation': signal_gen_results,
        'deep_learning': dl_results,
        'comparison_table': df.to_dict('records'),
        'metadata': {
            'python_version': sys.version,
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
    }
    
    with open(os.path.join(args.output_dir, 'combined_timing_results.pkl'), 'wb') as f:
        pickle.dump(combined_results, f)
    
    print("\n✓ Analysis complete! Generated files:")
    print(f"  - timing_comparison_best_combinations.csv")
    print(f"  - timing_comparison_best_combinations.png/pdf")
    print(f"  - speedup_comparison_best_combinations.png/pdf")
    print(f"  - timing_comparison_best_combinations_table.tex")
    print(f"  - combined_timing_results.pkl")
    
    # Print key findings
    best_dl_model = max(dl_results.items(), key=lambda x: x[1]['r2_score'])
    fastest_dl_model = min(dl_results.items(), key=lambda x: x[1]['avg_time'])
    
    best_speedup = signal_stats['mean'] / best_dl_model[1]['avg_time']
    fastest_speedup = signal_stats['mean'] / fastest_dl_model[1]['avg_time']
    
    print(f"\n=== KEY FINDINGS ===")
    print(f"Best performing model: {best_dl_model[0]} (R² = {best_dl_model[1]['r2_score']:.2f}, {best_speedup:.0f}× speedup)")
    print(f"Fastest model: {fastest_dl_model[0]} ({fastest_speedup:.0f}× speedup)")
    print(f"Physics-based simulation: {signal_stats['mean']:.2f}s per sample")
    print(f"Deep learning range: {min(r['avg_time'] for r in dl_results.values()):.6f}s to {max(r['avg_time'] for r in dl_results.values()):.6f}s per sample")
    
    return 0

if __name__ == "__main__":
    exit(main())