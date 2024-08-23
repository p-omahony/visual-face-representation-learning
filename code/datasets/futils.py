'''Module containing useful methods needed to run the pipeline.'''

from typing import Union

import numpy as np
import pandas as pd


def read_raw_data(
    txt_file_path: str,
    min_identities: Union[int, None] = None,
    max_identities: Union[int, None] = None
) -> None:
    """Read the txt file of faceSrub dataset to get the metadata.
    
    Args:
        txt_file_path: Path of the txt file.
        
        min_identities: From nth identity to download.
        
        max_identities: First n identities to download.
    
    Returns:
        final_df: Corresponding metadata needed to download images.
    """
    with open(txt_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    cols = lines[0].split('\t')[:-1]
    lines = [line.split('\t') for line in lines[1:]]

    data = {}
    for i, col in enumerate(cols):
        data[col] = np.array(lines)[:, i].tolist()

    df = pd.DataFrame(data)

    to_download = list(df['name'].value_counts()[:].index)

    if max_identities and min_identities:
        to_download = to_download[min_identities:max_identities]
    elif max_identities and not min_identities:
        to_download = to_download[:max_identities]
    elif min_identities and not max_identities:
        to_download = to_download[min_identities:]

    final_df = df[df['name'].isin(to_download)].reset_index(drop=True)

    return final_df
