import json
import shutil
from typing import Any

import torch
import os.path
from batchgenerators.utilities.file_and_folder_operations import save_pickle, load_pickle


__all__ = ['save_pickle', 'save_json', 'load_json', 'load_pickle', 'maybe_mkdir', 'empty_cache', 'dummy_context']




def load_json(file: Any):
    with open(file, 'r', encoding='utf-8') as f:
        a = json.load(f)
    return a


def save_json(obj, file, indent=4, sort_keys=False):
    with open(file, 'w') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent, ensure_ascii=False)


def maybe_mkdir(Dir, clean=False):
    if clean and os.path.isdir(Dir):
        shutil.rmtree(Dir)
    os.makedirs(Dir, exist_ok=True)


def empty_cache(device: torch.device):
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        from torch import mps
        mps.empty_cache()
    else:
        pass


class dummy_context(object):
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
