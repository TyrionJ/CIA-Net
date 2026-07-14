import os
import numpy as np
from os.path import join
from batchgenerators.dataloading.data_loader import DataLoader


class NetDataloader(DataLoader):
    def __init__(self, data_dir, selected, batch_size, in_chns, patch_size):
        super().__init__(None, batch_size, 1, None, True, False, True)

        self.data_dir = data_dir
        self.indices = self.collect_indices(selected)
        self.patch_size = patch_size
        self.img_shape = [batch_size, ] + [in_chns, ] + patch_size
        self.lbl_shape = [batch_size, ] + [2, ] + patch_size
        self.cache = {}

    def collect_indices(self, selected):
        data_keys = [f[:-4] for f in os.listdir(self.data_dir) if f.endswith('.pkl')]
        return sorted([k for k in data_keys if k in selected])

    def generate_train_batch(self):
        selected_keys = self.get_indices()

        image_all = np.zeros(self.img_shape, dtype=np.float32)
        label_all = np.zeros(self.lbl_shape, dtype=np.int16)

        for i, key in enumerate(selected_keys):
            if key in self.cache:
                data = self.cache[key]
            else:
                data = np.load(join(self.data_dir, f'{key}.npz'))
                self.cache[key] = data
            image, label = data['img'], data['seg']

            shape = image.shape[1:]
            bbox_lbs, bbox_ubs = get_bbox(shape, self.patch_size)

            data_slice = tuple([slice(0, image.shape[0])] + [slice(i, j) for i, j in zip(bbox_lbs, bbox_ubs)])
            image = image[data_slice]

            lbl_slice = tuple([slice(0, label.shape[0])] + [slice(i, j) for i, j in zip(bbox_lbs, bbox_ubs)])
            label = label[lbl_slice]

            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(3)]
            image_all[i] = np.pad(image, ((0, 0), *padding), 'constant', constant_values=0)
            label_all[i] = np.pad(label, ((0, 0), *padding), 'constant', constant_values=0)

        return {'image': image_all, 'label': label_all}


def get_bbox(shape, patch_size):
    lbs, ubs = [0, 0, 0], [max(i-j, 0) for i, j in zip(shape, patch_size)]
    bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(3)]
    bbox_ubs = [bbox_lbs[i] + patch_size[i] for i in range(3)]

    return bbox_lbs, bbox_ubs
