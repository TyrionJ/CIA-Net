import os
import shutil
import numpy as np
import nibabel as nb
from glob import glob
from tqdm import tqdm
from typing import Any
from sklearn.model_selection import KFold
from os.path import join, isfile, basename, exists
from batchgenerators.utilities.file_and_folder_operations import write_pickle

from utils.helpers import say_hi
from scripts.common import get_name_by_id
from utils.cropping import crop_to_nonzero
from utils.boundary_factory import get_boundaries
from utils.waiting_process import waiting_proc, mp
from utils.folder_file_operator import maybe_mkdir, load_json, save_json, save_pickle


class Processor:
    def __init__(self, raw_dir, processed_dir, dataset_id, images_dir='imagesTr', labels_dir='labelsTr', logger=print):
        super(Processor, self).__init__()

        self.cut_thr = 0.1
        self.dataset_name = get_name_by_id(raw_dir, dataset_id)
        self.raw_dataset: Any = join(raw_dir, self.dataset_name)
        self.processed_dataset: Any = join(processed_dir, self.dataset_name)
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.processed_image_dir = join(self.processed_dataset, images_dir)
        self.processed_label_dir = join(self.processed_dataset, labels_dir)
        self.processed_data_dir = join(self.processed_dataset, 'data')
        maybe_mkdir(self.processed_dataset)

        self.dataset = dataset = load_json(join(self.raw_dataset, 'dataset.json'))
        self.modalities = dataset['modalities'] if 'modalities' in dataset else dataset['channel_names']
        self.labels = sorted(list(dataset['labels'].values()))
        self.labels.remove(0)
        self.logger = logger or print
        say_hi(self.logger)

    @staticmethod
    def clip_normalization(data):
        mask = np.ones_like(data)
        mask[data == 0] = 0
        p995 = np.percentile(data[mask == 1], 99.5)
        p005 = np.percentile(data[mask == 1], 0.05)
        data = np.clip(data, p005, p995)
        data = (data - p005) / (p995 - p005)
        return data * mask

    def analyze_one(self, img_file, roi_file):
        """
        Analysis subject statistic for DDP
        :param img_file:
        :param roi_file:
        :return: Statistical data of each ROI on the image, including median, 0.5% and 99.5% percentile, mean, and std
        """
        img_data = nb.load(img_file).get_fdata()
        roi_data = nb.load(roi_file).get_fdata()
        img_data = self.clip_normalization(img_data)
        prop = {}
        for ROI in self.labels:
            values = img_data[roi_data == ROI]
            prop[ROI] = {
                'median': np.median(values),
                'percentile_99_5': np.percentile(values, 99.5),
                'percentile_00_5': np.percentile(values, 0.5),
                'mean': np.mean(values),
                'std': np.std(values),
            }
        return prop

    @staticmethod
    def cluster_domain(m_props, cut_thr):
        """
        Divide individual modal domains
        :param m_props: Properties of each modality
        :param cut_thr: domain similarity threshold
        :return: partitioned groups information
        """

        # Determine whether two domains belong to the same domain.
        in_one_group = lambda means, _i, _j: abs(means[_i] - means[_j]) / max(abs(means[_i]), abs(means[_j])) <= cut_thr

        # Initialize each ROI as a separate domain group.
        groups = [[v] for k, v in m_props.items()]
        groups = sorted(groups, key=lambda item: item[0]['median'])
        while True:
            # Use the median of each group as the partition criterion.
            group_means = [np.median([i['median'] for i in group]) for group in groups]
            # Check whether the partitioning is complete.
            stop = True
            for i in range(len(group_means)):
                for j in range(i + 1, len(group_means)):
                    # Two domain groups can be merged.
                    if in_one_group(group_means, i, j):
                        stop = False
                        break
                if not stop:
                    break
            if stop:
                break

            # Record statistics for the new grouping.
            new_groups, used = [], set()
            for i in range(len(groups)):
                if i in used:
                    continue
                new_group = groups[i].copy()
                for j in range(i + 1, len(groups)):
                    if j in used:
                        continue
                    # Two domain groups can be merged.
                    if in_one_group(group_means, i, j):
                        # Merge two groups
                        new_group.extend(groups[j])
                        used.add(j)
                new_groups.append(new_group)
                used.add(i)
            groups = new_groups

        return groups

    def props_split(self, data_props):
        """
        Main workflow for DDP computation process in one modality.
        :param data_props: Statistical results of all subjects in this modality across ROIs.
        :return:
        """

        # Record the statistical results of all data in this modality.
        m_prop = {}
        for data_prop in data_props:
            for ROI, props in data_prop.items():
                if ROI not in m_prop:
                    m_prop[ROI] = props
                else:
                    # Merge the median, quantiles, mean, and standard deviation for each ROI.
                    for k, v in props.items():
                        m_prop[ROI][k] += v
        prop_size = len(data_props)
        for ROI, props in m_prop.items():
            # Take the average of each statistical metric.
            for k, v in props.items():
                props[k] /= prop_size
            # Record the ROI label.
            props['label'] = ROI

        # Divide individual modal domains
        groups = self.cluster_domain(m_prop, self.cut_thr)

        # Organize the DDP results based on this modality’s partitioning outcome,
        # i.e., merge the ROIs and statistical information within each domain.
        info_groups = []
        for group in groups:
            group_info = {}
            for props in group:
                for k, v in props.items():
                    if k not in group_info:
                        group_info[k] = [v]
                    else:
                        group_info[k].append(v)
            for k, v in group_info.items():
                # Average the statistical results, while keeping the labels as the merged set of ROIs.
                if k != 'label':
                    group_info[k] = np.mean(group_info[k])
            info_groups.append(group_info)
        return info_groups

    def domain_split(self, img_keys):
        """
        Data domain partition
        :param img_keys: List[str]
        :return: domains information
        """
        self.logger('Domain spliting')

        domain_file = join(self.processed_dataset, 'domains.json')
        if isfile(domain_file):
            return load_json(domain_file)

        img_dir, lbl_dir = join(self.raw_dataset, self.images_dir), join(self.raw_dataset, self.labels_dir)
        domains = {}
        for modality in self.modalities.keys():
            m, rs = int(modality), []
            # Asynchronous processing of each subject
            with mp.get_context('spawn').Pool(12) as _p:
                for img_key in img_keys:
                    img_file = join(img_dir, f'{img_key}_{m:04d}.nii.gz')
                    roi_file = join(lbl_dir, f'{img_key}.nii.gz')
                    # analysis subject statistic
                    rs.append(_p.starmap_async(self.analyze_one, ((img_file, roi_file),)))
                waiting_proc(rs, _p, f'  split m-{modality}')
            m_props = [r.get()[0] for r in rs]
            domains[modality] = self.props_split(m_props)
        save_json(domains, domain_file)
        return domains

    @staticmethod
    def process_one(img_files, domains, roi_file, processed_image_dir='', processed_label_dir=''):
        """
        Preprocess single subject in all modalities
        :param img_files: paired images across all modalities
        :param domains: DDP result
        :param roi_file: label ROIS file
        :param processed_image_dir: save image folder
        :param processed_label_dir: save label folder
        :return: preprocessed images data
        """

        msk_data = nb.load(roi_file).get_fdata()
        msk_data[msk_data > 0] = 1
        if exists(processed_label_dir):
            shutil.copy(roi_file, join(processed_label_dir, basename(roi_file)))

        result, chn_idx = [], 0
        for n, img_file in enumerate(img_files):
            img_nii = nb.load(img_file)
            img_data = img_nii.get_fdata()
            img_data = Processor.clip_normalization(img_data)
            result.append(img_data)

            if exists(processed_image_dir):
                to_file = join(processed_image_dir, f'{basename(roi_file)[:-7]}_{chn_idx:04d}.nii.gz')
                nb.Nifti1Image(img_data, img_nii.affine).to_filename(to_file)
                chn_idx += 1

            for domain in domains[n]:
                p005 = domain['percentile_00_5']
                p995 = domain['percentile_99_5']
                data: Any = np.copy(img_data).clip(p005, p995) * msk_data
                result.append(data)

                if exists(processed_image_dir):
                    to_file = join(processed_image_dir, f'{basename(roi_file)[:-7]}_{chn_idx:04d}.nii.gz')
                    nb.Nifti1Image(data.astype(np.float32), img_nii.affine).to_filename(to_file)
                    chn_idx += 1

        return np.array(result)

    def process_data(self, img_keys, domains):
        """
        Preprocess all data
        :param img_keys: iamge keys
        :param domains: DDP result
        :return:
        """
        self.logger('Data processing')

        maybe_mkdir(self.processed_image_dir, clean=True)
        maybe_mkdir(self.processed_label_dir, clean=True)
        img_dir, lbl_dir = join(self.raw_dataset, self.images_dir), join(self.raw_dataset, self.labels_dir)

        rs, m_keys = [], sorted(self.modalities.keys())
        with mp.get_context('spawn').Pool(1) as _p:
            domains = [domains[k] for k in m_keys]
            for img_key in img_keys:
                img_files = [join(img_dir, f'{img_key}_{int(m):04d}.nii.gz') for m in m_keys]
                roi_file = join(lbl_dir, f'{img_key}.nii.gz')
                pros_img_dir = self.processed_image_dir
                pros_lbl_dir = self.processed_label_dir
                # Asynchronously process single subject in all modalities
                rs.append(_p.starmap_async(self.process_one, ((img_files, domains, roi_file, pros_img_dir, pros_lbl_dir),)))
            waiting_proc(rs, _p, f'  State')

    def save_net_info(self, domains):
        self.logger('Saving net info')
        in_chns, groups, group_chns = 0, len(domains), []
        for m, v in domains.items():
            in_chns += len(v) + 1
            group_chns.append(len(v) + 1)
        out_chns = len(self.labels) + 1
        net_info = {
            'in_chns': in_chns,
            'out_chns': out_chns,
            'groups': groups,
            'group_chns': group_chns,
            'not_only_largest': self.dataset['not_only_largest']
        }
        save_json(net_info, join(self.processed_dataset, 'net_info.json'))

        return net_info

    def gen_dataset(self, net_info):
        print('Generating dataset')
        dataset = load_json(join(self.raw_dataset, 'dataset.json'))
        new_channel_names = {}

        chn_idx = 0
        for g_id in range(net_info['groups']):
            if str(g_id) in dataset['channel_names']:
                chn_name = dataset['channel_names'][str(g_id)]
            else:
                chn_name = next(iter(dataset['channel_names'].values()))
            for chn_id in range(net_info['group_chns'][g_id]):
                new_channel_names[chn_idx] = f'{chn_name}-{chn_id}'
                chn_idx += 1
        dataset['channel_names'] = new_channel_names
        dataset['labels'] = dict(sorted(dataset['labels'].items(), key=lambda item: item[1]))

        save_json(dataset, join(self.processed_dataset, 'dataset.json'))

    def split_5fold(self, img_keys):
        split_file = join(self.processed_dataset, 'splits_final.json')
        self.logger(' Splitting dataset ...')

        if not exists(split_file):
            splits = []
            k_fold = KFold(n_splits=5, shuffle=True, random_state=20184)
            for i, (train_idx, test_idx) in enumerate(k_fold.split(img_keys)):
                train_keys = np.array(img_keys)[train_idx]
                test_keys = np.array(img_keys)[test_idx]
                splits.append({
                    'train': list(train_keys),
                    'val': list(test_keys)
                })
            save_json(splits, split_file)
            write_pickle(splits, split_file.replace('json', 'pkl'))

    def prepare_common(self, img_keys):
        self.split_5fold(img_keys)
        domains = self.domain_split(img_keys)
        net_info = self.save_net_info(domains)
        self.process_data(img_keys, domains)
        self.gen_dataset(net_info)

    def generate_boundaries(self, lbl_data):
        exclude_bodies = self.dataset['exclude_boundaries']
        boundaries = get_boundaries(lbl_data, exclude=exclude_bodies)
        return boundaries

    def prepare_cia_net(self, img_keys):
        maybe_mkdir(self.processed_data_dir)

        for img_key in tqdm(img_keys):
            images_nii = sorted(glob(join(self.processed_image_dir, f'{img_key}*.nii.gz')))
            label_nii = join(self.processed_label_dir, f'{img_key}.nii.gz')
            nii = nb.load(label_nii)

            img_data = [nb.load(f).get_fdata() for f in images_nii]
            img_data = np.array(img_data)
            lbl_data = nii.get_fdata()
            ori_shape = lbl_data.shape
            boundaries = self.generate_boundaries(lbl_data)
            lbl_data = np.array([lbl_data, boundaries])

            img_data, lbl_data, bbox = crop_to_nonzero(img_data, lbl_data, margin=5)
            pkl_info = {'crop_box': bbox, 'ori_shape': ori_shape, 'affine': nii.affine}

            np.savez(join(self.processed_data_dir, f'{img_key}.npz'), img=img_data, seg=lbl_data)
            save_pickle(pkl_info, join(self.processed_data_dir, f'{img_key}.pkl'))

    def run(self):
        self.logger(f'Preprocessing dataset {self.dataset_name} ...')

        img_keys = [i[:-7] for i in sorted(os.listdir(join(self.raw_dataset, self.labels_dir))) if i.endswith('.nii.gz')]
        img_keys = sorted(img_keys)[:1]

        self.prepare_common(img_keys)
        self.prepare_cia_net(img_keys)
