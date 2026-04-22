import os
import sys
import torch
import pickle
import numpy as np
from monai.transforms import Compose, ToTensor
from torch.utils.data import Dataset, Subset, DataLoader
import SimpleITK as sitk

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from architectures.Regressor import CustomRegressor
from architectures.DenseNet import DenseNet169
from tqdm import tqdm
from sklearn.metrics import r2_score
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="Find samples where both models perform well")
    parser.add_argument('--loss_fn', type=str, default='CustomL1Loss', 
                        help='Loss function used: L1Loss, MSELoss, CustomL1Loss, HuberLoss')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='Optimizer used: Adam, SGD, RMSprop, AdamW')
    parser.add_argument('--r2_threshold', type=float, default=0.95,
                        help='Minimum R² score to consider a sample good')
    parser.add_argument('--top_n', type=int, default=10,
                        help='Number of top common samples to return')
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
                key = (row_tuple[1], row_tuple[2], row_tuple[3])  # (b_value, small_delta, big_delta)
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

                    self.data.append({
                        'images': tiff_files,
                        'signals': signals,
                        'mri_params': mri_params,
                        'original_params': np.array([b, small_delta, big_delta], dtype=np.float32)
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

        return {
            "images": volume_4d,
            "mri_params": mri_params,
            "label": signals,
            "original_params": original_params
        }

    @staticmethod
    def load_3d_tiff(tiff_path):
        image = sitk.ReadImage(tiff_path)
        return sitk.GetArrayFromImage(image)

def custom_collate(batch):
    images = torch.stack([item['images'] for item in batch])
    mri_params = torch.stack([item['mri_params'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    original_params = torch.stack([item['original_params'] for item in batch])
    return {
        'images': images,
        'mri_params': mri_params,
        'label': labels,
        'original_params': original_params
    }

def load_model(architecture, loss_fn, optimizer):
    model_name = f"best_{architecture}_{loss_fn}_{optimizer}"
    models_path = f"models/{model_name}.pth"
    
    if architecture == "Regressor":
        model = CustomRegressor()
    elif architecture == "DenseNet169":
        model = DenseNet169(
            spatial_dims=3,
            in_channels=9,
            out_channels=11,
            init_features=64,
            growth_rate=32,
            dropout_prob=0.2
        )
    
    model.load_state_dict(torch.load(models_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def find_common_good_samples(args):
    # Load data
    with open(os.path.join('splits', 'data_list.pkl'), 'rb') as f:
        data_list = pickle.load(f)
    with open(os.path.join('splits', 'test_indices.pkl'), 'rb') as f:
        test_indices = pickle.load(f)

    test_transforms = Compose([ToTensor()])
    test_ds = TiffDataset(data_list, transform=test_transforms)
    test_dataset = Subset(test_ds, test_indices)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True,
        collate_fn=custom_collate
    )
    
    # Load models
    regressor_model = load_model("Regressor", args.loss_fn, args.optimizer)
    densenet_model = load_model("DenseNet169", args.loss_fn, args.optimizer)
    
    # Evaluate models on test set
    regressor_performances = []
    densenet_performances = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating models")):
            images = batch['images'].to(device)
            mri_params = batch['mri_params'].to(device)
            labels = batch['label'].to(device)
            
            # Get predictions
            regressor_preds = regressor_model(images, mri_params)
            densenet_preds = densenet_model(images, mri_params)
            
            for j in range(images.shape[0]):
                sample_idx = batch_idx * test_loader.batch_size + j
                original_idx = test_indices[sample_idx]
                
                # Calculate R² scores
                regressor_r2 = r2_score(
                    labels[j].cpu().numpy().flatten(), 
                    regressor_preds[j].cpu().numpy().flatten()
                )
                
                densenet_r2 = r2_score(
                    labels[j].cpu().numpy().flatten(), 
                    densenet_preds[j].cpu().numpy().flatten()
                )
                
                # Store sample index, test index, original index, and R² score
                regressor_performances.append((sample_idx, original_idx, regressor_r2))
                densenet_performances.append((sample_idx, original_idx, densenet_r2))
    
    # Find samples where both models performed well
    regressor_good_samples = set(idx for idx, _, r2 in regressor_performances if r2 >= args.r2_threshold)
    densenet_good_samples = set(idx for idx, _, r2 in densenet_performances if r2 >= args.r2_threshold)
    
    common_good_samples = regressor_good_samples.intersection(densenet_good_samples)
    
    # If no common samples found with the threshold, gradually lower it until we find some
    original_threshold = args.r2_threshold
    while len(common_good_samples) < args.top_n and args.r2_threshold > 0.5:
        args.r2_threshold -= 0.05
        regressor_good_samples = set(idx for idx, _, r2 in regressor_performances if r2 >= args.r2_threshold)
        densenet_good_samples = set(idx for idx, _, r2 in densenet_performances if r2 >= args.r2_threshold)
        common_good_samples = regressor_good_samples.intersection(densenet_good_samples)
        if len(common_good_samples) > 0:
            print(f"Found {len(common_good_samples)} common samples with R² >= {args.r2_threshold:.2f}")
    
    if len(common_good_samples) == 0:
        print(f"No common good samples found even with threshold lowered to 0.5")
        return []
    
    # Get detailed metrics for common good samples
    common_samples_info = []
    for idx in common_good_samples:
        # Find the regressor and densenet entries for this sample
        regressor_entry = next((entry for entry in regressor_performances if entry[0] == idx), None)
        densenet_entry = next((entry for entry in densenet_performances if entry[0] == idx), None)
        
        if regressor_entry and densenet_entry:
            # Calculate average R² score
            avg_r2 = (regressor_entry[2] + densenet_entry[2]) / 2
            
            # Get sample info
            test_sample = test_dataset[idx]
            original_idx = regressor_entry[1]  # Same as densenet_entry[1]
            b_value, small_delta, big_delta = test_sample['original_params'].numpy()
            
            # Get sample path (for identification)
            sample_path = '/'.join(data_list[original_idx]['images'][0].split('/')[-3:-1])
            
            common_samples_info.append({
                'test_idx': idx,
                'original_idx': original_idx,
                'sample_path': sample_path,
                'regressor_r2': regressor_entry[2],
                'densenet_r2': densenet_entry[2],
                'avg_r2': avg_r2,
                'b_value': b_value,
                'small_delta': small_delta,
                'big_delta': big_delta
            })
    
    # Sort by average R² score (descending)
    common_samples_info.sort(key=lambda x: x['avg_r2'], reverse=True)
    
    # Print results
    print(f"\nFound {len(common_samples_info)} samples where both models achieved R² >= {args.r2_threshold:.2f}")
    if original_threshold != args.r2_threshold:
        print(f"Note: Threshold was lowered from {original_threshold:.2f} to {args.r2_threshold:.2f}")
    
    print("\nTop common good samples:")
    for i, sample in enumerate(common_samples_info[:args.top_n]):
        print(f"\n{i+1}. Test Index: {sample['test_idx']} (Original Index: {sample['original_idx']})")
        print(f"   Path: {sample['sample_path']}")
        print(f"   MRI Params: b={sample['b_value']}, δ={sample['small_delta']}, Δ={sample['big_delta']}")
        print(f"   R² Scores: Regressor={sample['regressor_r2']:.4f}, DenseNet169={sample['densenet_r2']:.4f}, Avg={sample['avg_r2']:.4f}")
    
    return common_samples_info[:args.top_n]

if __name__ == "__main__":
    args = parse_args()
    find_common_good_samples(args)