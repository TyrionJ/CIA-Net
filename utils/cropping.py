import numpy as np
from scipy.ndimage import binary_dilation, binary_fill_holes
from acvl_utils.cropping_and_padding.bounding_boxes import get_bbox_from_mask, bounding_box_to_slice


def create_nonzero_mask(data):
    assert len(data.shape) == 4 or len(data.shape) == 3, "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
    nonzero_mask = np.zeros(data.shape[1:], dtype=bool)
    for c in range(data.shape[0]):
        this_mask = data[c] != 0
        nonzero_mask = nonzero_mask | this_mask
    nonzero_mask = binary_fill_holes(nonzero_mask)
    return nonzero_mask


def crop_to_nonzero(data, seg, margin=0):
    mask_seg = seg if len(seg.shape) == 3 else seg[0]
    nonzero_mask = np.ones_like(mask_seg, dtype=bool)
    nonzero_mask[mask_seg == 0] = False
    while margin > 0:
        nonzero_mask = binary_dilation(nonzero_mask, iterations=1)
        margin -= 1
    bbox = get_bbox_from_mask(nonzero_mask)

    slicer = bounding_box_to_slice(bbox)
    data = data[tuple([slice(None), *slicer])]

    if seg is not None:
        seg = seg[slicer] if len(seg.shape) == 3 else seg[tuple([slice(None), *slicer])]

    return data, seg, bbox
