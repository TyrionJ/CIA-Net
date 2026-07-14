import numpy as np
from tqdm import tqdm
from skimage.measure import label
from utils.boundary_factory import get_dilation


def keep_only_largest_component(segmentation: np.ndarray, foreground_labels, background_label: int = 0):
    mask = np.zeros_like(segmentation, dtype=bool)
    for reg in foreground_labels:
        mask |= region_or_label_to_mask(segmentation, reg)
    mask_keep = remove_all_but_largest_component(mask)
    ret = np.copy(segmentation)
    ret[mask & ~mask_keep] = background_label
    return ret


def remove_all_but_largest_component(binary_image: np.ndarray, connectivity: int = None):
    filter_fn = lambda x, y: [i for i, j in zip(x, y) if j == max(y)]
    return generic_filter_components(binary_image, filter_fn, connectivity)


def generic_filter_components(binary_image: np.ndarray, filter_fn, connectivity: int = None):
    labeled_image, component_sizes = label_with_component_sizes(binary_image, connectivity)
    component_ids = list(component_sizes.keys())
    component_sizes = list(component_sizes.values())
    keep = filter_fn(component_ids, component_sizes)
    return np.in1d(labeled_image.ravel(), keep).reshape(labeled_image.shape)


def label_with_component_sizes(binary_image: np.ndarray, connectivity: int = None):
    if not binary_image.dtype == bool:
        print('Warning: it would be way faster if your binary image had dtype bool')
    labeled_image, num_components = label(binary_image, return_num=True, connectivity=connectivity)
    component_sizes = {i + 1: j for i, j in enumerate(np.bincount(labeled_image.ravel())[1:])}
    return labeled_image, component_sizes


def region_or_label_to_mask(segmentation: np.ndarray, region_or_label):
    if np.isscalar(region_or_label):
        return segmentation == region_or_label
    else:
        mask = np.zeros_like(segmentation, dtype=bool)
        for r in region_or_label:
            mask[segmentation == r] = True
    return mask
