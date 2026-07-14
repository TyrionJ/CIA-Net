import os
import torch
import warnings
import numpy as np
import nibabel as nb
from tqdm import tqdm
from typing import Any
from os.path import join, isfile, dirname

from scripts.common import get_name_by_id
from scripts.preprocessor import Processor
from utils.helpers import say_hi, empty_cache
from utils.post_process import process_and_save
from utils.waiting_process import mp, waiting_proc
from utils.folder_file_operator import maybe_mkdir
from utils.cropping import bounding_box_to_slice, crop_to_nonzero
from utils.data_process_utils import logics_to_segmentation, get_sliding_window_slicers

warnings.filterwarnings("ignore")


class Predictor:
    group_chns = out_chns = patch_size = domains = not_only_largest = network = None

    def __init__(self, results_dir, dataset_id, task_name, net_class, folds,
                 input_dirs, output_dir, device, clear=False, logger=None):
        self.dataset_name: str = get_name_by_id(results_dir, dataset_id)
        self.result_task_dir: Any = join(results_dir, self.dataset_name, task_name)
        self.images_dir = input_dirs[0]
        self.masks_dir = join(dirname(input_dirs[0]), input_dirs[1])
        self.output_dir = output_dir
        self.net_class = net_class
        self.device = device \
            if isinstance(device, torch.device) \
            else torch.device('cpu') if device == 'cpu' \
            else torch.device(f'cuda:{device}')
        self.logger = logger or print
        self.use_model = 'model_final.pt'
        if folds == 'all':
            self.use_folds = [i for i in os.listdir(self.result_task_dir) if i.startswith('fold_')
                              and isfile(join(self.result_task_dir, i, self.use_model))]
        else:
            self.use_folds = [f'fold_{f}' for f in folds.split(',')]
        self.use_folds = sorted(self.use_folds)
        self.set_model_info()

        self.tile_step_size = 0.5
        self.clear = clear

        if output_dir:
            maybe_mkdir(output_dir)

    def set_model_info(self):
        info = torch.load(join(self.result_task_dir, self.use_folds[0], 'model_final.pt'), map_location='cpu')['info']
        self.group_chns = info['group_chns']
        self.out_chns = info['out_chns']
        self.patch_size = info['patch_size']
        self.not_only_largest = info['not_only_largest']
        self.domains = info['domains']
        self.network = self.net_class(self.group_chns, self.out_chns)

    def processed_generator(self):
        img_keys = sorted(list(set([i[:-12] for i in os.listdir(self.images_dir) if i.endswith('.nii.gz')])))
        self.logger(f'There are {len(img_keys)} case(s) to predict:\n')
        
        m_keys = sorted(self.domains.keys())
        for N, img_key in enumerate(img_keys):
            domains = [self.domains[k] for k in m_keys]
            img_files = [join(self.images_dir, f'{img_key}_{int(m):04d}.nii.gz') for m in m_keys]
            affine = nb.load(img_files[0]).affine
            img_data = Processor.process_one(img_files, domains, join(self.masks_dir, f'{img_key}.nii.gz'))
            msk_data = nb.load(join(self.masks_dir, f'{img_key}.nii.gz')).get_fdata()

            ori_shape = img_data[0].shape
            img_data, lbl_data, bbox = crop_to_nonzero(img_data, msk_data, margin=5)
            slicer = bounding_box_to_slice(bbox)
            msk_data = msk_data[slicer]

            yield torch.from_numpy(img_data).float(), img_key, (ori_shape, affine, slicer, msk_data), f'{N+1}/{len(img_keys)}'

    def logics_from_preprocessed(self, data):
        ultimate_prediction: Any = None
        with torch.no_grad():
            for fold in sorted(self.use_folds):
                model_info = torch.load(join(self.result_task_dir, fold, 'model_final.pt'), map_location='cpu')
                self.network.load_state_dict(model_info['weights'])
                self.network.to(self.device)

                predicted_logics = torch.zeros((self.out_chns, *data.shape[1:]), dtype=torch.float32,
                                               device=self.device)
                n_predictions = torch.zeros(data.shape[1:], dtype=torch.short, device=self.device)
                slicers = get_sliding_window_slicers(self.patch_size, data.shape[1:], self.tile_step_size)
                with torch.no_grad():
                    for sli in tqdm(slicers, desc=f'  In {fold}'):
                        workon = data[sli].to(self.device)[None]
                        prediction = self.network(workon)
                        predicted_logics[sli] += prediction[0]
                        n_predictions[sli[1:]] += 1
                        # empty_cache(self.device)
                predicted_logics /= n_predictions

                if ultimate_prediction is None:
                    ultimate_prediction = predicted_logics
                else:
                    ultimate_prediction += predicted_logics

        return ultimate_prediction / len(self.use_folds)

    def run(self, post=False):
        say_hi(self.logger)

        self.logger(f'Use fold(s): {self.use_folds}')
        self.logger(f'Model: {self.use_model}')
        self.logger(f'Patch size: {self.patch_size}')

        rs = []
        with mp.get_context('spawn').Pool(12) as _p:
            for data, img_key, meta, st in self.processed_generator():
                self.logger(f'[{st}] Predicting {img_key}:')
                to_file = join(self.output_dir, f'{img_key}.nii.gz')
                if os.path.exists(to_file):
                    self.logger(f'  skip {img_key}\n')
                    continue

                ori_shape, affine, slicer, msk_data = meta
                logics = self.logics_from_preprocessed(data)
                segm = logics_to_segmentation(logics, filter_labels=self.not_only_largest)

                ultimate: Any = np.zeros(ori_shape)
                ultimate[slicer] = segm * msk_data

                if post:
                    rs.append(_p.starmap_async(process_and_save, ((ultimate, affine, to_file),)))
                    self.logger(f'  send to backend\n')
                else:
                    nb.Nifti1Image(ultimate.astype(np.uint8), affine).to_filename(to_file)
                    self.logger(f'done with {img_key}\n')
                # empty_cache(self.device)

            if post:
                waiting_proc(rs, _p, 'Backend progress')
