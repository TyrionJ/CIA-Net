import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from scipy.ndimage import generate_binary_structure, binary_erosion, binary_dilation

from utils.waiting_process import waiting_proc


def get_surface(data: np.ndarray, connectivity=1, iterations=1):
    erosion = get_erosion(data, connectivity, iterations)
    return data - erosion


def get_erosion(data: np.ndarray, connectivity=1, iterations=1):
    data = np.atleast_1d(data.astype(bool))
    footprint = generate_binary_structure(data.ndim, connectivity)
    erosion = binary_erosion(data, structure=footprint, iterations=iterations)
    return erosion.astype(np.int16)


def get_dilation(data: np.ndarray, connectivity=1, iterations=1):
    data = np.atleast_1d(data.astype(bool))
    footprint = generate_binary_structure(data.ndim, connectivity)
    dilation = binary_dilation(data, structure=footprint, iterations=iterations)
    return dilation.astype(np.int16)


def process_body(data, idx):
    new_data = np.zeros_like(data)
    new_data[data == idx] = 1
    surface = get_surface(new_data)

    return surface


def get_boundaries(data: np.ndarray, desc=None, exclude=None):
    if exclude is None:
        exclude = []
    indices = sorted(list(set(data.flatten())))
    indices.remove(0)

    rs = []
    with mp.get_context('spawn').Pool(20) as p:
        for idx in indices:
            if idx in exclude:
                continue
            rs.append(p.starmap_async(process_body, ((data, idx),)))
        waiting_proc(rs, p, desc)
    bodies = [r.get()[0] for r in rs]

    return np.array(bodies).sum(axis=0)


def get_thin_bound(data: np.ndarray, desc=None):
    indices = sorted(list(set(data.flatten())))
    indices.remove(0)

    boundaries = np.zeros_like(data)
    for idx in tqdm(indices, disable=not desc, desc=desc):
        new_data = np.zeros_like(data)
        new_data[data == idx] = 1
        self_bound = get_surface(new_data)
        outer_bound = get_surface(get_dilation(new_data))

    return boundaries


def get_weak_boundaries(data: np.ndarray, desc=None):
    indices = sorted(list(set(data.flatten())))
    indices.remove(0)

    weak_boundaries = np.zeros_like(data)
    for idx in tqdm(indices, disable=not desc, desc=desc):
        new_data = np.zeros_like(data)
        new_data[data == idx] = 1
        w_idx = 0
        while True:
            erosion = get_erosion(new_data)
            if np.max(erosion) == 0:
                break
            weak_boundaries += (new_data - erosion) * (1 / np.power(2, w_idx))
            w_idx += 1
            new_data = erosion

    return weak_boundaries


if __name__ == '__main__':
    import nibabel as nb

    label_nii = nb.load(r'F:\Data\runtime\mT-Net\mT-Net_raw\Dataset001_QSMT1-EXT\labelsTr\QT_001.nii.gz')
    label_data = label_nii.get_fdata()

    bnd = get_boundaries(label_data[0], 'State')
    nb.Nifti1Image(bnd, label_nii.affine).to_filename('d:/boundaries.nii.gz')

    # w_bnd = get_weak_boundaries(label_data, 'State')
    # nb.Nifti1Image(w_bnd, label_nii.affine).to_filename('d:/weak_boundaries.nii.gz')
