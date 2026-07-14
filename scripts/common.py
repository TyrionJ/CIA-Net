import os
from os.path import isdir

def get_name_by_id(folder, ID):
    assert isdir(folder), f"The requested folder could not be found: {folder}"

    for name in os.listdir(folder):
        if f'{ID:03d}' in name:
            return name
    raise f'The requested {ID} could not be found in {folder}'
