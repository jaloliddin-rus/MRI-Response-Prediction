"""
Analyze TIFF Channel Value Ranges

This script scans all TIFF files in the dataset to determine the actual min/max
values for each input channel. This is useful for:
- Setting consistent color scales in ParaView
- Understanding the physical meaning of normalized values
- Identifying outliers or data quality issues

The 9 input channels are:
0. Binary (vessel mask)
1. SO2 (oxygen saturation)
2. Velocity (magnitude)
3-5. Vx, Vy, Vz (velocity components)
6-8. Distance fields (or other features)

Usage:
    python analyze_tiff_ranges.py [--sample_size N]
"""

import os
import json
import numpy as np
import pickle
import SimpleITK as sitk
from tqdm import tqdm
import argparse

def load_3d_tiff(tiff_path):
    """Load a 3D TIFF file"""
    image = sitk.ReadImage(tiff_path)
    return sitk.GetArrayFromImage(image)

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze value ranges in TIFF files")
    parser.add_argument('--sample_size', type=int, default=None,
                       help='Number of samples to analyze (default: all)')
    parser.add_argument('--percentile', type=float, default=99.9,
                       help='Percentile to report (default: 99.9 for ignoring extreme outliers)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load data list
    with open(os.path.join('splits', 'data_list.pkl'), 'rb') as f:
        data_list = pickle.load(f)
    
    # Sample if requested
    if args.sample_size is not None:
        import random
        data_list = random.sample(data_list, min(args.sample_size, len(data_list)))
    
    print(f"Analyzing {len(data_list)} samples...")
    print(f"Each sample has 9 TIFF channels\n")
    
    # Channel names (actual ordering from data_list.pkl)
    channel_names = [
        "Binary (vessel mask)",
        "Gradient",
        "Hct (hematocrit)",
        "PO2 (partial pressure O2, mmHg)",
        "SO2 (oxygen saturation, fraction)",
        "Velocity (magnitude, μm/s)",
        "Vx (velocity X, μm/s)",
        "Vy (velocity Y, μm/s)",
        "Vz (velocity Z, μm/s)"
    ]
    
    # Initialize tracking arrays
    num_channels = 9
    global_min = [np.inf] * num_channels
    global_max = [-np.inf] * num_channels
    global_mean = [0.0] * num_channels
    global_std = [0.0] * num_channels
    percentile_values = [[] for _ in range(num_channels)]
    nonzero_counts = [0] * num_channels
    voxel_counts = [0] * num_channels
    
    # Analyze each sample
    for item in tqdm(data_list, desc="Processing samples"):
        tiff_files = item['images']
        
        if len(tiff_files) != num_channels:
            print(f"Warning: Expected {num_channels} channels, got {len(tiff_files)}")
            continue
        
        # Load all channels for this sample
        for channel_idx, tiff_path in enumerate(tiff_files):
            if not os.path.exists(tiff_path):
                print(f"Warning: File not found: {tiff_path}")
                continue
            
            # Load volume
            volume = load_3d_tiff(tiff_path)
            
            # Update statistics
            global_min[channel_idx] = min(global_min[channel_idx], volume.min())
            global_max[channel_idx] = max(global_max[channel_idx], volume.max())
            global_mean[channel_idx] += volume.mean()
            global_std[channel_idx] += volume.std()

            # Track sparsity (useful to detect all-zero binary masks)
            nz = np.count_nonzero(volume)
            nonzero_counts[channel_idx] += nz
            voxel_counts[channel_idx] += volume.size
            
            # Store percentile
            percentile_values[channel_idx].append(np.percentile(volume, args.percentile))
    
    # Average the means and stds
    num_samples = len(data_list)
    global_mean = [m / num_samples for m in global_mean]
    global_std = [s / num_samples for s in global_std]
    
    # Calculate percentile across all samples
    global_percentile = [np.max(vals) if vals else 0 for vals in percentile_values]
    
    # Print results
    print(f"\n{'='*80}")
    print(f"TIFF Channel Value Ranges (analyzed {num_samples} samples)")
    print(f"{'='*80}\n")
    
    for channel_idx in range(num_channels):
        print(f"Channel {channel_idx}: {channel_names[channel_idx]}")
        print(f"  Global Min:     {global_min[channel_idx]:12.6f}")
        print(f"  Global Max:     {global_max[channel_idx]:12.6f}")
        print(f"  {args.percentile}th percentile: {global_percentile[channel_idx]:12.6f}")
        print(f"  Mean:           {global_mean[channel_idx]:12.6f}")
        print(f"  Std Dev:        {global_std[channel_idx]:12.6f}")
        if voxel_counts[channel_idx] > 0:
            sparsity = 1 - (nonzero_counts[channel_idx] / voxel_counts[channel_idx])
            print(f"  Non-zero voxels: {nonzero_counts[channel_idx]} / {voxel_counts[channel_idx]} (sparsity {sparsity:0.4f})")
        # Quick warning for masks that seem empty
        if channel_idx == 0 and nonzero_counts[channel_idx] == 0:
            print("  WARNING: Binary channel appears all-zero across analyzed samples.")
            print("           If masks are stored as 0/255, values are still detected; this suggests empty masks.")
        print()
    
    # Save to file (human-readable)
    output_txt = "results/tiff_channel_ranges.txt"
    os.makedirs("results", exist_ok=True)
    
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(f"TIFF Channel Value Ranges\n")
        f.write(f"Analyzed {num_samples} samples\n")
        f.write(f"{'='*80}\n\n")
        
        for channel_idx in range(num_channels):
            f.write(f"Channel {channel_idx}: {channel_names[channel_idx]}\n")
            f.write(f"  Global Min:     {global_min[channel_idx]:12.6f}\n")
            f.write(f"  Global Max:     {global_max[channel_idx]:12.6f}\n")
            f.write(f"  {args.percentile}th percentile: {global_percentile[channel_idx]:12.6f}\n")
            f.write(f"  Mean:           {global_mean[channel_idx]:12.6f}\n")
            f.write(f"  Std Dev:        {global_std[channel_idx]:12.6f}\n\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("Recommended ParaView Color Scale Ranges:\n")
        f.write("="*80 + "\n\n")
        
        for channel_idx in range(num_channels):
            # For binary, force 0-1 (expected mask range), fallback to observed min/max if outside
            if channel_idx == 0:
                scale_min = 0.0
                scale_max = 1.0
                if global_max[channel_idx] > 1 or global_min[channel_idx] < 0:
                    scale_min = global_min[channel_idx]
                    scale_max = global_max[channel_idx]
            else:
                # For others, use percentile to avoid extreme outliers
                scale_min = 0 if global_min[channel_idx] >= 0 else global_min[channel_idx]
                scale_max = global_percentile[channel_idx]
            
            f.write(f"Channel {channel_idx} ({channel_names[channel_idx]}):\n")
            f.write(f"  Recommended range: [{scale_min:.6f}, {scale_max:.6f}]\n\n")
    
    # Save machine-readable JSON for downstream visualization scripts
    output_json = "results/tiff_channel_ranges.json"
    channel_stats = {}
    for channel_idx in range(num_channels):
        if channel_idx == 0:
            # Prefer canonical mask range, but keep observed if outside 0-1
            rec_min = 0.0
            rec_max = 1.0
            if global_max[channel_idx] > 1 or global_min[channel_idx] < 0:
                rec_min = global_min[channel_idx]
                rec_max = global_max[channel_idx]
        else:
            rec_min = 0 if global_min[channel_idx] >= 0 else global_min[channel_idx]
            rec_max = global_percentile[channel_idx]
        channel_stats[channel_names[channel_idx]] = {
            "channel_index": channel_idx,
            "global_min": float(global_min[channel_idx]),
            "global_max": float(global_max[channel_idx]),
            "percentile": float(global_percentile[channel_idx]),
            "mean": float(global_mean[channel_idx]),
            "std": float(global_std[channel_idx]),
            "recommended_min": float(rec_min),
            "recommended_max": float(rec_max)
        }
    with open(output_json, "w", encoding="utf-8") as jf:
        json.dump(channel_stats, jf, indent=2)
    
    print(f"Results saved to: {output_txt}")
    print(f"JSON saved to:    {output_json}")
    print("\nRecommended ParaView color scale ranges:")
    print("="*80)
    
    for channel_idx in range(num_channels):
        if channel_idx == 0:
            scale_min = 0.0
            scale_max = 1.0
            if global_max[channel_idx] > 1 or global_min[channel_idx] < 0:
                scale_min = global_min[channel_idx]
                scale_max = global_max[channel_idx]
        else:
            scale_min = 0 if global_min[channel_idx] >= 0 else global_min[channel_idx]
            scale_max = global_percentile[channel_idx]
        
        print(f"Channel {channel_idx} ({channel_names[channel_idx]}): [{scale_min:.6f}, {scale_max:.6f}]")

if __name__ == "__main__":
    main()
