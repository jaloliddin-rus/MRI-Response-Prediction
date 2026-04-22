"""
Dataset Index Verification Tool

This script verifies that test indices correspond to the correct animal/chunk samples
by comparing what both structure_performance_analyzer.py and plot.py see.

Usage:
    python scripts/verify_indices.py --indices 239,153,69,27,35
"""

import os
import numpy as np
import torch
import argparse
import pickle
from monai.transforms import Compose, ToTensor
from torch.utils.data import Dataset, Subset
import SimpleITK as sitk
from sklearn.metrics import r2_score
import pandas as pd
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from architectures.DenseNet import DenseNet169

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="Verify test indices correspond to correct samples")
    parser.add_argument('--indices', type=str, required=True,
                       help='Comma-separated list of test indices to verify (e.g., 239,153,69)')
    parser.add_argument('--architecture', type=str, default='DenseNet169',
                       help='Architecture to test with')
    parser.add_argument('--loss_fn', type=str, default='HuberLoss',
                       help='Loss function')
    parser.add_argument('--optimizer', type=str, default='Adam',
                       help='Optimizer')
    return parser.parse_args()

class TiffDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = []
        self.transform = transform

        # Define normalization ranges
        self.b_range = [50, 500]
        self.small_delta_range = [1, 2]
        self.big_delta_range = [4, 7]

        # Predefined phi and theta combinations
        self.phi_theta_list = [
            (0, 0), (0, 90), (45, 45), (45, 90), (45, 135),
            (90, 45), (90, 90), (90, 135), (135, 45), (135, 90), (135, 135),
        ]

        for item in data:
            tiff_files = item['images']
            npy_file = item['label']
            npy_data = np.load(npy_file, allow_pickle=True)

            if npy_data.ndim == 0:
                npy_data = npy_data.item()

            # Group data by b_value, small_delta, and big_delta
            grouped_data = {}
            for row in npy_data:
                row_tuple = tuple(row.item())
                key = (row_tuple[1], row_tuple[2], row_tuple[3])
                if key not in grouped_data:
                    grouped_data[key] = []
                grouped_data[key].append(row_tuple)

            for (b, small_delta, big_delta), group in grouped_data.items():
                signals = []
                for phi, theta in self.phi_theta_list:
                    found = False
                    for row in group:
                        if row[4] == phi and row[5] == theta:
                            signals.append(row[0])
                            found = True
                            break
                    if not found:
                        break
                else:
                    # All phi-theta pairs were found
                    signals = np.array(signals, dtype=np.float32)

                    # Normalize parameters
                    norm_b = (b - self.b_range[0]) / (self.b_range[1] - self.b_range[0])
                    norm_small_delta = (small_delta - self.small_delta_range[0]) / (self.small_delta_range[1] - self.small_delta_range[0])
                    norm_big_delta = (big_delta - self.big_delta_range[0]) / (self.big_delta_range[1] - self.big_delta_range[0])

                    # Store MRI parameters as scalar values
                    mri_params = np.array([norm_b, norm_small_delta, norm_big_delta], dtype=np.float32)

                    # Extract animal/chunk info
                    sample_path = tiff_files[0]
                    path_parts = sample_path.replace('\\', '/').split('/')
                    animal_chunk = "unknown/unknown"
                    if len(path_parts) >= 3:
                        animal = path_parts[-3]
                        chunk = path_parts[-2]
                        animal_chunk = f"{animal}/{chunk}"

                    self.data.append({
                        'images': tiff_files,
                        'signals': signals,
                        'mri_params': mri_params,
                        'original_params': np.array([b, small_delta, big_delta], dtype=np.float32),
                        'animal_chunk': animal_chunk,
                        'tiff_files': tiff_files  # Store for verification
                    })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        if not hasattr(self, 'cached_images'):
            self.cached_images = {}
        image_key = tuple(item['images'])
        if image_key not in self.cached_images:
            volumes = [self.load_3d_tiff(img) for img in item['images']]
            if self.transform:
                volumes = [self.transform(vol) for vol in volumes]
            volume_4d = np.stack(volumes, axis=0).astype(np.float32)
            self.cached_images[image_key] = volume_4d
        else:
            volume_4d = self.cached_images[image_key]

        volume_4d = torch.from_numpy(volume_4d)
        signals = torch.from_numpy(item['signals']).float()
        mri_params = torch.from_numpy(item['mri_params']).float()
        original_params = torch.from_numpy(item['original_params']).float()
        animal_chunk = item['animal_chunk']

        return {
            "images": volume_4d,
            "mri_params": mri_params,
            "label": signals,
            "original_params": original_params,
            "animal_chunk": animal_chunk
        }

    @staticmethod
    def load_3d_tiff(tiff_path):
        image = sitk.ReadImage(tiff_path)
        return sitk.GetArrayFromImage(image)

def main():
    args = parse_args()
    
    # Parse indices
    indices = [int(idx.strip()) for idx in args.indices.split(',')]
    
    print("="*80)
    print("DATASET INDEX VERIFICATION TOOL")
    print("="*80)
    print(f"\nVerifying {len(indices)} test indices: {indices}")
    print(f"Model: {args.architecture} | Loss: {args.loss_fn} | Optimizer: {args.optimizer}\n")
    
    # Load data
    with open(os.path.join('splits', 'data_list.pkl'), 'rb') as f:
        data_list = pickle.load(f)
    with open(os.path.join('splits', 'test_indices.pkl'), 'rb') as f:
        test_indices = pickle.load(f)
    
    print(f"Loaded {len(data_list)} total samples")
    print(f"Test set has {len(test_indices)} samples\n")
    
    # Create dataset (same as both scripts)
    test_transforms = Compose([ToTensor()])
    test_ds = TiffDataset(data_list, transform=test_transforms)
    test_dataset = Subset(test_ds, test_indices)
    
    print(f"Dataset created with {len(test_ds.data)} processed samples")
    print(f"Test subset has {len(test_dataset)} samples\n")
    
    # Load model for R² verification
    model_path = f"models/best_{args.architecture}_{args.loss_fn}_{args.optimizer}.pth"
    print(f"Loading model: {model_path}")
    
    if args.architecture == "DenseNet169":
        model = DenseNet169(
            spatial_dims=3,
            in_channels=9,
            out_channels=11,
            init_features=64,
            growth_rate=32,
            dropout_prob=0.2
        )
    else:
        print(f"Architecture {args.architecture} not implemented in this script")
        return
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model = model.to(device)
    print("Model loaded successfully\n")
    
    print("="*80)
    print("VERIFICATION RESULTS")
    print("="*80)
    
    # Verify each index
    results = []
    for test_idx in indices:
        if test_idx >= len(test_dataset):
            print(f"\n❌ ERROR: Index {test_idx} out of range (max: {len(test_dataset)-1})")
            continue
        
        # Get sample from dataset
        sample = test_dataset[test_idx]
        
        # Get the corresponding entry in the full dataset
        original_idx = test_indices[test_idx]
        full_data_entry = test_ds.data[original_idx]
        
        animal_chunk = sample['animal_chunk']
        b_value = sample['original_params'][0].item()
        small_delta = sample['original_params'][1].item()
        big_delta = sample['original_params'][2].item()
        
        # Get file paths
        tiff_files = full_data_entry['tiff_files']
        first_file = tiff_files[0]
        
        # Run model prediction
        images = sample['images'].unsqueeze(0).to(device)
        mri_params = sample['mri_params'].unsqueeze(0).to(device)
        labels = sample['label']
        
        with torch.no_grad():
            prediction = model(images, mri_params).cpu()
        
        r2 = r2_score(labels.numpy().flatten(), prediction.numpy().flatten())
        
        # Store result
        result = {
            'test_idx': test_idx,
            'original_idx': original_idx,
            'animal_chunk': animal_chunk,
            'b_value': b_value,
            'small_delta': small_delta,
            'big_delta': big_delta,
            'r2': r2,
            'first_tiff': first_file
        }
        results.append(result)
        
        # Print detailed info
        print(f"\n{'─'*80}")
        print(f"Test Index: {test_idx}")
        print(f"{'─'*80}")
        print(f"  Animal/Chunk:     {animal_chunk}")
        print(f"  MRI Parameters:   b={b_value:.1f}, δ={small_delta:.1f}, Δ={big_delta:.1f}")
        print(f"  R² Score:         {r2:.4f}")
        print(f"  Original Index:   {original_idx} (position in data_list.pkl)")
        print(f"  First TIFF file:  {first_file}")
        print(f"\n  ✓ Use this for plot.py:")
        print(f"    python scripts/plot.py --architecture {args.architecture} --loss_fn {args.loss_fn} --optimizer {args.optimizer} --index {test_idx}")
        print(f"\n  ✓ For ParaView, open these TIFF files:")
        print(f"    {os.path.dirname(first_file)}/")
    
    # Create summary CSV
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print("="*80)
    
    df = pd.DataFrame(results)
    print(df[['test_idx', 'animal_chunk', 'b_value', 'small_delta', 'big_delta', 'r2']].to_string(index=False))
    
    # Save to file
    output_file = "results/verified_indices.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✓ Full verification results saved to: {output_file}")
    
    # Generate ParaView paths file
    paraview_file = "results/paraview_paths.txt"
    with open(paraview_file, 'w', encoding='utf-8') as f:
        f.write("ParaView TIFF Directory Paths\n")
        f.write("="*80 + "\n\n")
        for result in results:
            tiff_dir = os.path.dirname(result['first_tiff'])
            f.write(f"Test Index {result['test_idx']}: {result['animal_chunk']}\n")
            f.write(f"  MRI: b={result['b_value']:.1f}, δ={result['small_delta']:.1f}, Δ={result['big_delta']:.1f}\n")
            f.write(f"  R²: {result['r2']:.4f}\n")
            f.write(f"  Path: {tiff_dir}\n\n")
    
    print(f"✓ ParaView paths saved to: {paraview_file}")
    
    print(f"\n{'='*80}")
    print("VERIFICATION COMPLETE")
    print("="*80)
    print("\n✅ All indices verified successfully!")
    print("✅ You can now safely use these indices with plot.py")
    print("✅ TIFF directories listed for ParaView visualization")

if __name__ == "__main__":
    main()
