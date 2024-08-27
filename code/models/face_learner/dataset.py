from pathlib import Path

import numpy as np
import zarr
from PIL import Image
from torch.utils.data import Dataset


class FaceScrub(Dataset):
    def __init__(self, data_path, train=True, transforms=None) -> None:
        self.data_path = Path(data_path)
        self.labels = zarr.open(self.data_path / Path("labels.zarr"), "r")[:]
        self.ims = zarr.open(self.data_path / Path("ims.zarr"), "r")

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
            else:
                rdm_idx = np.random.choice(negatives_ims_indices)
                comp_im = self.ims[rdm_idx]

            if self.transforms:
                comp_im = self.transforms(comp_im)

            return anchor_im, comp_im, target

        return anchor_im


class FaceScrubTriplet(Dataset):
    def __init__(self, data_path, train=True, transforms=None) -> None:
        self.data_path = Path(data_path)
        self.labels = zarr.open(self.data_path / Path("labels.zarr"), "r")[:]
        self.ims = zarr.open(self.data_path / Path("ims.zarr"), "r")[:]

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

        if self.train:
            rdm_positive_idx = np.random.choice(positives_ims_indices)
            positive_im = self.ims[rdm_positive_idx]

            rdm_negative_idx = np.random.choice(negatives_ims_indices)
            negative_im = self.ims[rdm_negative_idx]

            if self.transforms:
                positive_im = self.transforms(positive_im)
                negative_im = self.transforms(negative_im)

            return anchor_im, positive_im, negative_im

        return anchor_im


class Amigos(Dataset):
    def __init__(self, images_array_path, transforms=None):
        self.images_array_path = images_array_path
        self.ims = zarr.open(self.images_array_path / Path("val_ims.zarr"), "r")[:]
        self.labels = zarr.open(
            self.images_array_path / Path("val_labels.zarr"), "r"
        )[:]

        self.transforms = transforms

    def __len__(self):
        return len(self.raw_images)

    def __getitem__(self, idx):
        im = self.ims[idx]
        label = self.labels[idx]
        if self.transforms:
            im = self.transforms(im)

        return im, label
