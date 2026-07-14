import os
import numpy as np
import nibabel as nb
from tqdm import tqdm
from os.path import join
from skimage.measure import label
from scipy.ndimage import generate_binary_structure, binary_dilation


def label_with_component_sizes(binary_image: np.ndarray, connectivity: int = None):
    if not binary_image.dtype == bool:
        print('Warning: it would be way faster if your binary image had dtype bool')
    labeled_image, num_components = label(binary_image, return_num=True, connectivity=connectivity)
    component_sizes = {i + 1: j for i, j in enumerate(np.bincount(labeled_image.ravel())[1:])}
    return labeled_image, component_sizes


def get_dilation(data: np.ndarray, connectivity=1, iterations=1):
    data = np.atleast_1d(data.astype(bool))
    footprint = generate_binary_structure(data.ndim, connectivity)
    dilation = binary_dilation(data, structure=footprint, iterations=iterations)
    return dilation.astype(np.int16)


def unique(data_arr, filters=None):
    if filters is None:
        filters = [91, 92, 100]
    eles, cnts = np.unique(data_arr, return_counts=True)
    rst = []
    for ele, cnt in zip(eles, cnts):
        rst.append([ele, cnt])

    rst2 = [i for i in rst if i[0] not in filters]
    if len(rst2) > 0:
        rst = rst2
    return sorted(rst, key=lambda x: x[1], reverse=True)


def process_and_save(seg_data, affine, save_path, desc=''):
    need_post = np.setdiff1d(list(range(1, 97)) + [100] + list(range(110, 116)), [53, 54])

    for old_label in tqdm(need_post, disable=not desc, desc=desc):
        mask = np.zeros_like(seg_data, dtype=bool)
        mask[seg_data == old_label] = 1
        labeled_img, labeled_sizes = label_with_component_sizes(mask, connectivity=1)
        labeled_sizes = sorted(labeled_sizes.items(), key=lambda item: item[1], reverse=True)
        for val, cnt in labeled_sizes[1:]:
            mask = np.zeros_like(seg_data, dtype=bool)
            mask[labeled_img == val] = 1
            mask = get_dilation(mask, connectivity=1) - mask
            around = seg_data[mask == 1]
            unis = unique(around)
            if len(unis) > 0:
                seg_data[labeled_img == val] = int(unis[0][0])

    nb.Nifti1Image(seg_data.astype(np.uint8), affine).to_filename(save_path)


def main():
    # src_dir = r'F:\Data\runtime\CIA-Net\CIA-Net_raw\Dataset002_QSMT1-Ext\Predictions\CIA-Net_nopost'
    # to_dir = r'F:\Data\runtime\CIA-Net\CIA-Net_raw\Dataset002_QSMT1-Ext\Predictions\CIA-Net_afterPost'
    src_dir = '/remote-home/hejj/Data/runtime/CIA-Net/CIA-Net_raw/Dataset002_QSMT1-Ext/Predictions/CIA-Net_nopost'
    to_dir = '/remote-home/hejj/Data/runtime/CIA-Net/CIA-Net_raw/Dataset002_QSMT1-Ext/Predictions/CIA-Net_afterPost'
    # src_dir = r'F:\Data\runtime\CIA-Net\CIA-Net_raw\Dataset001_QSMT1-120\Predictions\CIA-Net\Origin'
    # to_dir = r'F:\Data\runtime\CIA-Net\CIA-Net_raw\Dataset001_QSMT1-120\Predictions\CIA-Net\Origin_Post'
    os.makedirs(to_dir, exist_ok=True)

    for nii_file in sorted(os.listdir(src_dir))[22:]:
        seg_file = join(src_dir, nii_file)
        to_file = join(to_dir, nii_file)
        if os.path.exists(to_file):
            continue

        seg_nii = nb.load(seg_file)
        seg_data = seg_nii.get_fdata()
        process_and_save(seg_data, seg_nii.affine, to_file, desc=nii_file)


if __name__ == '__main__':
    main()
