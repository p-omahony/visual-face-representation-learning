from pathlib import Path

import numpy as np
import zarr
from torch.utils.data import Dataset


class FaceScrub(Dataset):
    def __init__(self, data_path, train=True, transforms=None) -> None:
        self.data_path = Path(data_path)
        self.labels = zarr.open(self.data_path / Path('labels.zarr'), 'r')[:]
        self.ims = zarr.open(self.data_path / Path('ims.zarr'), 'r')

        self.train = train
        self.transforms = transforms

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        anchor_im = self.ims[idx]
        anchor_label = self.labels[idx]

        positives_ims_indices = np.where(self.labels == anchor_label)[0]
        negatives_ims_indices = np.where(self.labels != anchor_label)[0]

        if self.transforms:
            anchor_im = self.transforms(anchor_im)

        target = np.random.randint(0, 2)

        if self.train:
            if target == 0:
                rdm_idx = np.random.choice(positives_ims_indices)
                comp_im = self.ims[rdm_idx]
            else :
                rdm_idx = np.random.choice(negatives_ims_indices)
                comp_im = self.ims[rdm_idx]

            if self.transforms:
                comp_im = self.transforms(comp_im)

            return anchor_im, comp_im, target

        return anchor_im
