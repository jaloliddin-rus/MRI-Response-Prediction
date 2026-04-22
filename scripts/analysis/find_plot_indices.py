"""
Find the correct test index in plot.py's dataset for a given animal/chunk
"""
import os
import sys
import pickle
from monai.transforms import Compose, ToTensor
from torch.utils.data import Subset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Force UTF-8 stdout so the non-ASCII characters below (delta, arrow, etc.)
# don't crash under Windows' default cp1252 console encoding.
try:
    sys.stdout.reconfigure(encoding='utf-8')
except (AttributeError, OSError):
    pass

from src.dataset import TiffDataset

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
    test_ds = TiffDataset(data_list, transform=test_transforms, include_animal_chunk=True)
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
