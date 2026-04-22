"""
Find the correct test index in plot.py's dataset for a given animal/chunk
"""
import os
import sys
import numpy as np
import pickle
from monai.transforms import Compose, ToTensor
from torch.utils.data import Dataset, Subset
import SimpleITK as sitk

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Force UTF-8 stdout so the non-ASCII characters below (delta, arrow, etc.)
# don't crash under Windows' default cp1252 console encoding.
try:
    sys.stdout.reconfigure(encoding='utf-8')
except (AttributeError, OSError):
    pass

# Use plot.py's TiffDataset (with filtering)
class TiffDataset(Dataset):
    def __init__(self, data, b_value=None, small_delta=None, big_delta=None, transform=None):
        self.data = []
        self.transform = transform

        self.b_range = [50, 500]
        self.small_delta_range = [1, 2]
        self.big_delta_range = [4, 7]

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

            # Filter data if MRI parameters are specified
            if b_value is not None and small_delta is not None and big_delta is not None:
                filtered_data = [row for row in npy_data if row[1] == b_value and row[2] == small_delta and row[3] == big_delta]
            else:
                filtered_data = npy_data

            grouped_data = {}
            for row in filtered_data:
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
                    signals = np.array(signals, dtype=np.float32)
                    norm_b = (b - self.b_range[0]) / (self.b_range[1] - self.b_range[0])
                    norm_small_delta = (small_delta - self.small_delta_range[0]) / (self.small_delta_range[1] - self.small_delta_range[0])
                    norm_big_delta = (big_delta - self.big_delta_range[0]) / (self.big_delta_range[1] - self.big_delta_range[0])
                    mri_params = np.array([norm_b, norm_small_delta, norm_big_delta], dtype=np.float32)

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
                        'animal_chunk': animal_chunk
                    })

    def __len__(self):
        return len(self.data)

    @staticmethod
    def load_3d_tiff(tiff_path):
        image = sitk.ReadImage(tiff_path)
        return sitk.GetArrayFromImage(image)

# Target samples from structure_performance_analyzer
targets = {
    "K3/chunk_536": (500.0, 1.0, 4.0),
    "K3/chunk_135": (50.0, 1.5, 5.0),
    "K3/chunk_46": (50.0, 1.5, 5.0),
    "K3/chunk_392": (50.0, 1.0, 4.0),
    "K3/chunk_346": (500.0, 2.0, 7.0),
    "K3/chunk_510": (500.0, 2.0, 7.0),
    "K3/chunk_337": (50.0, 2.0, 7.0),
    "K3/chunk_461": (50.0, 1.5, 5.0),
    "K3/chunk_1022": (500.0, 2.0, 7.0),
    "K3/chunk_472": (500.0, 2.0, 7.0),
    "K3/chunk_468": (500.0, 2.0, 7.0),
    "K3/chunk_371": (500.0, 2.0, 7.0),
    "K3/chunk_547": (500.0, 1.0, 4.0),
    "K3/chunk_153": (500.0, 2.0, 7.0),
}

def main():
    with open(os.path.join('splits', 'data_list.pkl'), 'rb') as f:
        data_list = pickle.load(f)
    with open(os.path.join('splits', 'test_indices.pkl'), 'rb') as f:
        test_indices = pickle.load(f)

    test_transforms = Compose([ToTensor()])
    test_ds = TiffDataset(data_list, transform=test_transforms)
    test_dataset = Subset(test_ds, test_indices)

    print("Finding correct indices for plot.py...")
    print("=" * 80)

    for target_chunk, (b, sd, bd) in targets.items():
        found = False
        for test_idx in range(len(test_dataset)):
            original_idx = test_indices[test_idx]
            sample_data = test_ds.data[original_idx]

            if (sample_data['animal_chunk'] == target_chunk and
                abs(sample_data['original_params'][0] - b) < 0.1 and
                abs(sample_data['original_params'][1] - sd) < 0.1 and
                abs(sample_data['original_params'][2] - bd) < 0.1):
                print(f"{target_chunk} (b={b}, delta={sd}, Delta={bd})")
                print(f"  -> Test index: {test_idx}")
                print(f"  Command: python scripts/visualization/plot.py "
                      f"--architecture DenseNet169 --loss_fn HuberLoss "
                      f"--optimizer Adam --index {test_idx}\n")
                found = True
                break

        if not found:
            print(f"NOT FOUND: {target_chunk} (b={b}, delta={sd}, Delta={bd})\n")


if __name__ == "__main__":
    main()
