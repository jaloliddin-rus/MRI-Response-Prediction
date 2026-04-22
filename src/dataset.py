"""TiffDataset — single canonical definition used by training and analysis.

Previously duplicated across `training.py`, `scripts/validation.py`, and most
analysis scripts. Keep this one copy in sync; every consumer should
`from src.dataset import TiffDataset, custom_collate`.
"""

import os
import numpy as np
import torch
import SimpleITK as sitk
from torch.utils.data import Dataset


# Acquisition-parameter normalization ranges and the fixed (phi, theta)
# directions the model expects. The TiffDataset only yields samples whose
# `signals_journal.npy` contains every one of these directions.
B_RANGE = (50.0, 500.0)
SMALL_DELTA_RANGE = (1.0, 2.0)
BIG_DELTA_RANGE = (4.0, 7.0)

PHI_THETA_LIST = [
    (0, 0), (0, 90), (45, 45), (45, 90), (45, 135),
    (90, 45), (90, 90), (90, 135), (135, 45), (135, 90), (135, 135),
]


class TiffDataset(Dataset):
    """Dataset of 3D vascular structures + diffusion signals.

    Each sample is one (structure, MRI-params) combination; the structure is
    a 9-channel 3D volume loaded from TIFF files, the label is the stack of
    11 diffusion-signal time-series (one per gradient direction).
    """

    def __init__(
        self,
        data,
        transform=None,
        b_value=None,
        small_delta=None,
        big_delta=None,
        include_animal_chunk=False,
    ):
        """
        Parameters
        ----------
        data : list[dict]
            Entries with keys 'images' (list of TIFF paths) and 'label'
            (path to signals_journal.npy).
        transform : callable or None
            Applied to each loaded 3D volume.
        b_value, small_delta, big_delta : float or None
            If all three are provided, only signal rows matching those
            acquisition parameters are kept. Useful for plotting a single
            (b, δ, Δ) condition.
        include_animal_chunk : bool
            If True, each stored sample gets an extra 'animal_chunk' field
            (e.g. "K3/chunk_0") derived from the first TIFF path. Needed by
            scripts that map dataset indices back to source structures.
        """
        self.data = []
        self.transform = transform

        self.b_range = B_RANGE
        self.small_delta_range = SMALL_DELTA_RANGE
        self.big_delta_range = BIG_DELTA_RANGE
        self.phi_theta_list = PHI_THETA_LIST

        filter_active = (
            b_value is not None and small_delta is not None and big_delta is not None
        )

        for item in data:
            tiff_files = item['images']
            npy_file = item['label']
            npy_data = np.load(npy_file, allow_pickle=True)

            if npy_data.ndim == 0:
                npy_data = npy_data.item()

            if filter_active:
                rows = [
                    r for r in npy_data
                    if r[1] == b_value and r[2] == small_delta and r[3] == big_delta
                ]
            else:
                rows = npy_data

            grouped_data = {}
            for row in rows:
                row_tuple = tuple(row.item())
                key = (row_tuple[1], row_tuple[2], row_tuple[3])  # (b, δ, Δ)
                grouped_data.setdefault(key, []).append(row_tuple)

            for (b, sd, bd), group in grouped_data.items():
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
                    norm_small_delta = (sd - self.small_delta_range[0]) / (
                        self.small_delta_range[1] - self.small_delta_range[0]
                    )
                    norm_big_delta = (bd - self.big_delta_range[0]) / (
                        self.big_delta_range[1] - self.big_delta_range[0]
                    )
                    mri_params = np.array(
                        [norm_b, norm_small_delta, norm_big_delta], dtype=np.float32
                    )

                    sample = {
                        'images': tiff_files,
                        'signals': signals,
                        'mri_params': mri_params,
                        'original_params': np.array(
                            [b, sd, bd], dtype=np.float32
                        ),
                    }

                    if include_animal_chunk:
                        sample['animal_chunk'] = _animal_chunk_from_tiff(tiff_files[0])

                    self.data.append(sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]

        # Per-worker volume cache: structures are reused across (b, δ, Δ)
        # combinations, so caching the 4D stack avoids re-reading 9 TIFFs.
        if not hasattr(self, 'cached_images'):
            self.cached_images = {}
        image_key = tuple(item['images'])
        if image_key not in self.cached_images:
            volumes = [self.load_3d_tiff(img) for img in item['images']]
            if self.transform:
                volumes = [self.transform(vol) for vol in volumes]
            volume_4d = np.array(volumes)
            self.cached_images[image_key] = volume_4d
        else:
            volume_4d = self.cached_images[image_key]

        volume_4d = torch.from_numpy(volume_4d).float()
        signals = torch.from_numpy(item['signals']).float()
        mri_params = torch.from_numpy(item['mri_params']).float()
        original_params = torch.from_numpy(item['original_params']).float()

        out = {
            "images": volume_4d,
            "mri_params": mri_params,
            "label": signals,
            "original_params": original_params,
        }
        if 'animal_chunk' in item:
            out['animal_chunk'] = item['animal_chunk']
        return out

    @staticmethod
    def load_3d_tiff(tiff_path):
        image = sitk.ReadImage(tiff_path)
        return sitk.GetArrayFromImage(image)


def _animal_chunk_from_tiff(tiff_path):
    """Return "<animal>/<chunk>" for a path like `data/K3/chunk_0/binary.tiff`."""
    parts = tiff_path.replace('\\', '/').split('/')
    if len(parts) >= 3:
        return f"{parts[-3]}/{parts[-2]}"
    return "unknown/unknown"


def custom_collate(batch):
    """Stack dict-valued samples into dict-of-tensors batches.

    If the samples include an 'animal_chunk' string (present when the dataset
    was built with include_animal_chunk=True), it is returned as a list under
    'animal_chunks'.
    """
    out = {
        'images': torch.stack([item['images'] for item in batch]),
        'mri_params': torch.stack([item['mri_params'] for item in batch]),
        'label': torch.stack([item['label'] for item in batch]),
        'original_params': torch.stack([item['original_params'] for item in batch]),
    }
    if batch and 'animal_chunk' in batch[0]:
        out['animal_chunks'] = [item['animal_chunk'] for item in batch]
    return out
