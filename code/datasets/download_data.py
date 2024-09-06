'''Module containing methods to download faceScrub dataset's images.'''

import os
from pathlib import Path
from typing import List

import cv2
import requests
from pandas import DataFrame
from PIL import Image
from tqdm import tqdm

from futils import read_raw_data

REQUEST_HEADER = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}

def download_image(url: str, save_path: str, im_id: str, im_ext: str) -> bool:
    """Downloads an image from url."""
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    try:
        response = requests.get(url, headers=REQUEST_HEADER, timeout=0.5)
        if response.status_code == 200:
            with open(f'{save_path}/{im_id}{im_ext}', 'wb') as file:
                file.write(response.content)
            return True
    except:
        print(f'Could not download image at: {url}')
    return False


def download_images(save_root_dir: str, download_data: DataFrame):
    """Download images."""
    rows = list(download_data.iterrows())
    downloaded_images_save_paths = []
    for row in tqdm(rows):
        name = row[1]['name'].strip().replace(' ', '_')
        image_id: str = row[1]['image_id']
        url = row[1]['url']
        ext = Path(url).suffix
        save_path = os.path.join(save_root_dir, name)

        successfully_downloaded = download_image(url, save_path, image_id, ext)
        if successfully_downloaded:
            downloaded_images_save_paths.append(f'{save_path}/{image_id}{ext}')

    return downloaded_images_save_paths


def clean_downloaded_images(downloaded_images: List[str]):
    """Deletes corrupted images. -> UGLY, TO CHANGE."""
    for im_path in tqdm(downloaded_images):
        im = cv2.imread(str(im_path))
        try:
            shape = im.shape
        except:
            os.remove(str(im_path))

        try:
            img = Image.open(str(im_path))
            img.load()  # Force loading the image
        except:
            if os.path.exists(str(im_path)):
                os.remove(str(im_path))


def main(min_identities, max_identities):
    """Run download pipeline."""
    # download_data = read_raw_data(
    #     './data/raw/faceScrub/facescrub_actors.txt',
    #     min_identities,
    #     max_identities
    # )
    # print('Downloading images...')
    # downloaded_images = download_images(
    #     './data/processed/face_scrub',
    #     download_data
    # )
    # print('Download completed successfully!')
    downloaded_images = [f for f in Path('./data/processed/face_scrub').rglob("*") if f.is_file()]
    print('Cleaning downloaded images...')
    clean_downloaded_images(downloaded_images)
    print('Clean completed successfully!')


if __name__ == '__main__':
    main(200, 201)
