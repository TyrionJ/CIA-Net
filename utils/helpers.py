import torch
import random
import numpy as np


class dummy_context(object):
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def empty_cache(device: torch.device):
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    else:
        pass


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def say_hi(logger):
    logger("\n#######################################################################\n"
           "Please cite the following paper when using CIA-Net:\n\n"
           "#######################################################################\n")
