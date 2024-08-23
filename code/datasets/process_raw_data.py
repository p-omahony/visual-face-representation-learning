'''Module used to process downloaded faceScrub's images.'''

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import zarr
from numpy.typing import NDArray
from pandas import DataFrame
from PIL import Image
from tqdm import tqdm

from futils import read_raw_data

IM_SIZE = 224

def get_face(images_root_path: str, image_metadata: DataFrame):
    """Crops and transforms the face from image given image metadata."""
    im_id = image_metadata['image_id']
    ext = Path(image_metadata['url']).suffix
    bbox = [int(e) for e in image_metadata['bbox'].split(',')]
    label = image_metadata['name'].strip().replace(' ', '_')

    im_path = os.path.join(images_root_path, label, im_id, ext)
    im = Image.open(im_path)
    im_transformed = im.crop(bbox).resize((IM_SIZE, IM_SIZE)).convert('RGB')

    return (
        im_transformed,
        label
    )


def get_downloaded_images_metadata(data_root_path: str, metadata_df: DataFrame):
    """Get only downloaded images metadata."""
    images = [im for im in Path(data_root_path).rglob('*') if im.is_file()]
    imids = [im.stem.replace('.', '') for im in images]
    imlabels = [im.parent.name.replace('_', ' ') for im in images]
    imdata = list(zip(imids, imlabels))

    metadata_df['imdata'] = pd.Series(
        list(zip(metadata_df['image_id'], metadata_df['name']))
    )
    metadata_df = metadata_df[
        metadata_df['imdata'].isin(imdata)
    ].reset_index(drop=True)

    return metadata_df


def extract_faces(metadata_df: DataFrame) -> Tuple[NDArray, NDArray]:
    """Extract faces from all images for which the metadata is given."""
    identities = metadata_df['name'].unique()
    faces, labels = [], []
    for identity in tqdm(identities):
        identity_data = metadata_df[metadata_df['name'] == identity]
        for row in identity_data.iterrows():
            face, _ = get_face('../data/processed/face_scrub', row[1])
            faces.append(np.array(face))
            labels.append(identity.encode('utf-8'))
    faces = np.array(faces)
    labels = np.array(labels)

    return faces, labels

def create_dataset(faces_ims: NDArray, labels, save_path: NDArray) -> None:
    """Creates and saves a dataset as a zarr array."""
    z_labels = zarr.open(
        os.path.join(save_path, 'labels.zarr'),
        mode='w',
        shape=labels.shape,
        chunks=(len(labels),),
        dtype='S100'
    )
    z_ims = zarr.open(
        os.path.join(save_path, 'ims.zarr'),
        mode='w',
        shape=faces_ims.shape,
        chunks=(1, 224, 224, 3),
        dtype='uint8'
    )

    z_labels[:], z_ims[:] = z_labels, faces_ims


def main():
    """"Runs processing pipeline."""
    final_df = read_raw_data(
        './data/raw/faceScrub/facescrub_actors.txt',
    )
    final_df = get_downloaded_images_metadata(
        './data/processed/face_scrub',
        final_df
    )
    faces, labels = extract_faces(final_df)
    create_dataset(faces, labels, save_path='')


if __name__ == '__main__':
    main()
