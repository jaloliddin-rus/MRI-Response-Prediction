"""
Inspect actual TIFF file structure to understand channel organization
"""

import os
import pickle
import SimpleITK as sitk
import numpy as np

def load_3d_tiff(tiff_path):
    image = sitk.ReadImage(tiff_path)
    return sitk.GetArrayFromImage(image)

# Load data list
with open(os.path.join('splits', 'data_list.pkl'), 'rb') as f:
    data_list = pickle.load(f)

# Take first sample
sample = data_list[0]
print("First sample:")
print(f"Number of TIFF files: {len(sample['images'])}")
print()

# Check each TIFF file
for idx, tiff_path in enumerate(sample['images']):
    print(f"Channel {idx}:")
    print(f"  Path: {tiff_path}")
    
    if os.path.exists(tiff_path):
        volume = load_3d_tiff(tiff_path)
        print(f"  Shape: {volume.shape}")
        print(f"  Min: {volume.min():.6f}")
        print(f"  Max: {volume.max():.6f}")
        print(f"  Mean: {volume.mean():.6f}")
        print(f"  Non-zero values: {np.count_nonzero(volume)} / {volume.size} ({100*np.count_nonzero(volume)/volume.size:.2f}%)")
        
        # Check filename for hints
        filename = os.path.basename(tiff_path)
        print(f"  Filename: {filename}")
    else:
        print(f"  ERROR: File not found!")
    print()
