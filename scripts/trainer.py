import time
import torch
import os.path
import warnings
import numpy as np
import nibabel as nb
from tqdm import tqdm
from typing import List, Any
from datetime import datetime
from os.path import join, isdir
from torch import any, tensor, isnan
from torch.cuda.amp import GradScaler
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter

from scripts.common import get_name_by_id
from scripts.dataloader import NetDataloader
from utils.evaluation import get_tp_fp_fn_tn
from utils.collate_outputs import collate_outputs
from utils.polyrescheduler import PolyLRScheduler
from utils.default_n_proc import get_allowed_n_proc
from utils.helpers import empty_cache, say_hi, set_seed
from transforms.train_transform import TrTransform, VdTransform
from transforms.limited_length_multithreaded_augmenter import LimitedLenWrapper
from utils.folder_file_operator import load_json, load_pickle, save_json, maybe_mkdir
from utils.data_process_utils import logics_to_segmentation, get_sliding_window_slicers

warnings.filterwarnings('ignore')
set_seed()


class NetTrainer:
    group_chns = out_chns = not_only_largest = None
    network = optimizer = lr_scheduler = None
    processed_dir = result_dataset = fold_dir = result_task_dir = final_valid_dir = None

    def __init__(self, processed_dir, result_dir, dataset_id, task_name, batch_size, patch_size, net_class, loss_fn,
                 fold, go_on, epochs, device, validation=False, logger=print):

        self.patch_size = patch_size
        self.batch_size = batch_size
        self.fold = fold
        self.go_on = go_on
        self.epochs = epochs
        self.validation = validation
        self.dataset_id = dataset_id
        self.result_dir = result_dir
        self.task_name = task_name
        self.net_class = net_class
        self.device = torch.device(f'cuda:{device}') if device != 'cpu' else torch.device(device)

        self.install_folder(processed_dir)
        self.save_model_info()
        self.logger = self.build_logger(logger)
        say_hi(self.logger)

        self.cur_epoch = 0
        self.initial_lr = 1e-2
        self.weight_decay = 3e-5
        self.train_iters = 250
        self.valid_iters = 50
        self.save_interval = 2
        self.best_dice = 0

        self.network = self.net_class(self.group_chns, self.out_chns).to(self.device)
        self.train_loader, self.valid_loader = self.get_tr_vd_loader()
        self.grad_scaler = GradScaler() if device != 'cpu' else None
        self.loss_fn = loss_fn

    def build_logger(self, logger):
        now = datetime.now()
        prefix = 'training' if not self.validation else 'validation'
        log_file = join(self.fold_dir, f'{prefix}_log_{now.strftime("%Y-%m-%d_%H-%M-%S")}.txt')
        fw = open(log_file, 'a', encoding='utf-8')

        def log_fn(content):
            logger(content)
            fw.write(f'{content}\n')
            fw.flush()

        return log_fn

    def install_folder(self, processed_dir):
        assert 0 <= self.fold < 5, 'only support 5-fold training, and fold should belong to [0, 5)'
        assert isdir(processed_dir), "The requested processed data folder could not be found"

        d_name = get_name_by_id(processed_dir, self.dataset_id)
        self.processed_dir: Any = join(processed_dir, d_name)
        self.result_task_dir: Any = join(self.result_dir, d_name, self.task_name)
        self.fold_dir = join(self.result_task_dir, f'fold_{self.fold}')
        self.final_valid_dir = join(self.fold_dir, 'validation')
        maybe_mkdir(self.final_valid_dir)

        net_info = load_json(join(self.processed_dir, 'net_info.json'))
        self.group_chns = net_info['group_chns']
        self.out_chns = net_info['out_chns']
        self.not_only_largest = net_info['not_only_largest']

    def save_model_info(self):
        info = {
            'group_chns': self.group_chns,
            'out_chns': self.out_chns,
            'patch_size': self.patch_size,
            'domains': load_json(join(self.processed_dir, 'domains.json')),
            'not_only_largest': self.not_only_largest
        }
        save_json(info, join(self.result_task_dir, 'model_info.json'))

    def get_tr_vd_indices(self, verbose=True):
        s_file = join(self.processed_dir, 'splits_final.json')
        splits = load_json(s_file)
        fold = splits[self.fold]

        if verbose:
            self.logger(f'Use splits: {s_file}')
            self.logger(f'Training patch size: {self.patch_size}')
            self.logger(f'The file contains {len(splits)} splits.')
            self.logger(f'Fold for training: {self.fold}')
            self.logger(f'Patch size: {self.patch_size}')
        return fold['train'], fold['val']

    def get_tr_vd_loader(self):
        train_indices, valid_indices = self.get_tr_vd_indices()
        self.logger(f"tr_set size={len(train_indices)}, val_set size={len(valid_indices)}")
        data_fdr = join(self.processed_dir, 'data')
        in_chns = sum(self.group_chns)
        tr_loader = NetDataloader(data_fdr, train_indices, self.batch_size, in_chns, self.patch_size)
        vd_loader = NetDataloader(data_fdr, valid_indices, max(2, self.batch_size // 2), in_chns, self.patch_size)
        tr_transforms, val_transforms = TrTransform(self.network.sup_depth), VdTransform(self.network.sup_depth)

        allowed_num_processes = get_allowed_n_proc()
        if allowed_num_processes == 0 or self.device.type == 'cpu':
            mt_gen_train = SingleThreadedAugmenter(tr_loader, tr_transforms)
            mt_gen_val = SingleThreadedAugmenter(vd_loader, val_transforms)
        else:
            mt_gen_train = LimitedLenWrapper(self.train_iters, data_loader=tr_loader,
                                             transform=tr_transforms,
                                             num_processes=allowed_num_processes, num_cached=6, seeds=None,
                                             pin_memory=self.device.type == 'cuda')
            mt_gen_val = LimitedLenWrapper(self.valid_iters, data_loader=vd_loader,
                                           transform=val_transforms, num_processes=max(1, allowed_num_processes // 2),
                                           num_cached=3, seeds=None, pin_memory=self.device.type == 'cuda')
            time.sleep(0.1)
        return mt_gen_train, mt_gen_val

    def initialize(self):
        empty_cache(self.device)
        self.logger(f'Supervision depth: {self.network.sup_depth}')
        self.loss_fn = self.loss_fn(self.network.sup_depth)
        self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                         momentum=0.99, nesterov=True)
        self.lr_scheduler = PolyLRScheduler(self.optimizer, self.initial_lr, self.epochs)
        self.load_states()

    def load_states(self):
        check_file = join(self.fold_dir, 'model_latest.pt')
        if self.go_on:
            if os.path.isfile(check_file):
                self.logger(f'Use checkpoint: {check_file}')
                weights = torch.load(join(self.fold_dir, 'model_latest.pt'), map_location=torch.device('cpu'))
                checkpoint = torch.load(join(self.fold_dir, 'check_latest.pth'), map_location=torch.device('cpu'))

                if 'cur_epoch' in weights:
                    del weights['cur_epoch']
                self.network.load_state_dict(weights)
                self.optimizer.load_state_dict(checkpoint['optimizer_state'])
                self.cur_epoch = checkpoint['cur_epoch']
                self.best_dice = checkpoint['best_dice']
                if self.grad_scaler is not None and checkpoint['grad_scaler_state'] is not None:
                    self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])
            else:
                self.logger('No checkpoint found, start a new training')

    def save_states(self, val_dice):
        self.cur_epoch += 1
        checkpoint = {
            'optimizer_state': self.optimizer.state_dict(),
            'cur_epoch': self.cur_epoch,
            'best_dice': self.best_dice,
            'grad_scaler_state': self.grad_scaler.state_dict() if self.grad_scaler is not None else None
        }
        if self.grad_scaler is not None:
            checkpoint['grad_scaler_state'] = self.grad_scaler.state_dict()

        if self.best_dice < val_dice:
            self.best_dice = val_dice
            self.logger(f'Eureka!!! Best dice: {self.best_dice:.4f}')
            torch.save(self.network.state_dict(), join(self.fold_dir, 'model_best.pt'))

        if self.cur_epoch % self.save_interval == 0 or self.cur_epoch == self.epochs:
            torch.save(checkpoint, join(self.fold_dir, 'check_latest.pth'))
            torch.save(self.network.state_dict(), join(self.fold_dir, 'model_latest.pt'))
        self.logger('')

    def train_step(self, batch: dict):
        img_data = batch['image']
        lbl_data = batch['label']

        img_data = img_data.to(self.device)
        if isinstance(lbl_data, list):
            lbl_data = [i.to(self.device, ) for i in lbl_data]
        else:
            lbl_data = lbl_data.to(self.device)

        self.optimizer.zero_grad()

        net_out = self.network(img_data, **{'epoch': self.cur_epoch})
        if any(tensor([any(tensor([isnan(m).any() for m in net_out[i]])) for i in range(len(net_out))])):
            return None
        t_loss = self.loss_fn(net_out, lbl_data)

        try:
            if self.grad_scaler is not None:
                self.grad_scaler.scale(t_loss).backward()
                self.grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                t_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()
        except RuntimeError as e:
            print(e)
            return None
        return {'loss': t_loss.item()}

    def valid_step(self, batch: dict):
        img_data = batch['image']
        lbl_data = batch['label']

        img_data = img_data.to(self.device)
        if isinstance(lbl_data, list):
            lbl_data = [i.to(self.device) for i in lbl_data]
        else:
            lbl_data = lbl_data.to(self.device)

        net_out = self.network(img_data)
        if any(tensor([any(tensor([isnan(m).any() for m in net_out[i]])) for i in range(len(net_out))])):
            return None
        t_loss = self.loss_fn(net_out, lbl_data)

        seg_out = net_out[0][0] if isinstance(net_out[0], list) else net_out[0]
        seg_tgt = lbl_data[0][:, :1] if isinstance(lbl_data, list) else lbl_data[:, :1]

        axes = [0] + list(range(2, len(seg_out.shape)))
        output_seg = seg_out.argmax(1)[:, None]
        predicted_segmentation_onehot = torch.zeros(seg_out.shape, device=seg_out.device, dtype=torch.float32)
        predicted_segmentation_onehot.scatter_(1, output_seg, 1)
        del output_seg

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, seg_tgt, axes=axes)

        tp_hard = tp.detach().cpu().numpy()[1:]
        fp_hard = fp.detach().cpu().numpy()[1:]
        fn_hard = fn.detach().cpu().numpy()[1:]

        return {'loss': t_loss.item(),
                'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}

    def on_train_epoch_end(self, epoch, train_loss, lr):
        self.logger(f'Epoch: {epoch} / {self.epochs}')
        self.logger(f'current lr: {np.round(lr, decimals=5)}')
        self.logger(f'train loss: {np.round(train_loss, decimals=6)}')

    def on_valid_epoch_end(self, val_outputs: List[dict]):
        val_outputs = list(filter(None, val_outputs))
        outputs_collated = collate_outputs(val_outputs)
        tp = np.sum(outputs_collated['tp_hard'], 0)
        fp = np.sum(outputs_collated['fp_hard'], 0)
        fn = np.sum(outputs_collated['fn_hard'], 0)

        loss_here = np.mean(outputs_collated['loss']).astype(np.float64)

        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k + 1e-8) for i, j, k in zip(tp, fp, fn)]]
        mean_fg_dice = np.nanmean(global_dc_per_class)

        self.logger(f'validation loss: {np.round(loss_here, decimals=6)}')
        self.logger(f'valid mean Dice: {np.round(mean_fg_dice, decimals=6)}')
        self.logger(f'valid best Dice: {np.round(np.maximum(self.best_dice, mean_fg_dice), decimals=6)}')
        self.logger(f'vDice per class: [{", ".join([f"{i:.2f}" for i in global_dc_per_class])}]')

        return mean_fg_dice

    def save_final_model(self):
        final_check = {
            'weights': torch.load(join(self.fold_dir, 'model_best.pt')),
            'info': {
                'group_chns': self.group_chns,
                'out_chns': self.out_chns,
                'patch_size': self.patch_size,
                'domains': load_json(join(self.processed_dir, 'domains.json')),
                'not_only_largest': self.not_only_largest
            }
        }
        torch.save(final_check, join(self.fold_dir, 'model_final.pt'))

    def conduct_final_validation(self):
        self.logger('\nFinal Validation')

        self.network.deep_supervision = False
        if self.validation:
            weights = torch.load(join(self.fold_dir, 'model_best.pt'), map_location=self.device)
            self.network.load_state_dict(weights)
            print('Load best model')

        obj = {}
        mean_Dice = 0
        train_indices, valid_indices = self.get_tr_vd_indices(False)
        self.logger(f'There {len(valid_indices)} case(s) to validate:\n')
        for N, key in enumerate(valid_indices):
            self.logger(f'[{N+1}/{len(valid_indices)}] Validating {key}:')
            obj[key] = {}

            npz_data = np.load(join(self.processed_dir, 'data', f'{key}.npz'))
            pkl_info = load_pickle(join(self.processed_dir, 'data', f'{key}.pkl'))

            in_data = torch.from_numpy(npz_data['img']).float()
            predicted_logics = torch.zeros((self.out_chns, *in_data.shape[1:]), dtype=torch.half, device=self.device)
            n_predictions = torch.zeros(in_data.shape[1:], dtype=torch.half, device=self.device)
            slicers = get_sliding_window_slicers(self.patch_size, in_data.shape[1:])

            with torch.no_grad():
                for sli in tqdm(slicers, desc='  State'):
                    workon = in_data[sli].to(self.device)
                    prediction = self.network(workon[None])
                    predicted_logics[sli] += prediction[0]
                    n_predictions[sli[1:]] += 1
                    # empty_cache(self.device)

            predicted_logics /= n_predictions
            pred_segm = logics_to_segmentation(predicted_logics, filter_labels=self.not_only_largest)

            segm_onehot = torch.zeros((1, self.out_chns) + pred_segm.shape, dtype=torch.float32)
            segm_onehot.scatter_(1, torch.from_numpy(pred_segm[None, None].astype(np.int64)), 1)
            tp, fp, fn, _ = get_tp_fp_fn_tn(segm_onehot, torch.from_numpy(npz_data['seg'][0:1]), axes=(0, 2, 3, 4))
            tp = tp.numpy()[1:]
            fp = fp.numpy()[1:]
            fn = fn.numpy()[1:]
            DCpC = [i for i in [2 * i / (2 * i + j + k + 1e-8) for i, j, k in zip(tp, fp, fn)]]
            m_DC = np.mean(DCpC)
            mean_Dice = (mean_Dice * N + m_DC) / (N + 1)

            self.logger(f'  AvDC={np.round(m_DC, decimals=6)}')
            self.logger(f'  DCpC={[np.round(i, decimals=2) for i in DCpC]}')

            obj[key]['m_DC'] = m_DC
            obj[key]['DCpC'] = DCpC

            to_file = join(self.final_valid_dir, f'{key}.nii.gz')
            sli = tuple([slice(i, j) for i, j in pkl_info['crop_box']])
            original_seg: Any = np.zeros(pkl_info['ori_shape'])
            original_seg[sli] = pred_segm
            nb.Nifti1Image(original_seg, pkl_info['affine']).to_filename(to_file)
            self.logger(f'  Prediction saved\n')

        self.logger('Final Validation complete')
        self.logger(f'  Mean Validation Dice: {np.round(mean_Dice, decimals=6)}')
        obj['mean_dice'] = mean_Dice
        save_json(obj, join(self.fold_dir, 'validation_summary.json'))

    def run(self):
        self.initialize()

        if not self.validation:
            self.logger('\nBegin training ...')
            time.sleep(0.5)

            self.network.deep_supervision = True
            for epoch in range(self.cur_epoch, self.epochs):
                avg_loss = 0

                self.lr_scheduler.step(self.cur_epoch)
                lr = self.optimizer.param_groups[0]['lr']

                self.network.train()
                with tqdm(desc=f'[{epoch + 1}/{self.epochs}]Training', total=self.train_iters) as p:
                    for batch_id in range(self.train_iters):
                        train_loss = self.train_step(next(self.train_loader))
                        train_loss = train_loss['loss'] if train_loss is not None else avg_loss
                        avg_loss = (avg_loss * batch_id + train_loss) / (batch_id + 1)
                        p.set_postfix(**{'avg': '%.4f' % avg_loss, 'bat': '%.4f' % train_loss, 'lr': '%.6f' % lr,
                                         'T': self.task_name, 'f': self.fold})
                        p.update()

                self.network.eval()
                with torch.no_grad():
                    with tqdm(desc='~~Validation', total=self.valid_iters, colour='green') as p:
                        val_outputs = []
                        for batch_id in range(self.valid_iters):
                            val_outputs.append(self.valid_step(next(self.valid_loader)))
                            p.update()

                self.on_train_epoch_end(epoch + 1, avg_loss, lr)
                val_dice = self.on_valid_epoch_end(val_outputs)
                self.save_states(val_dice)
            self.logger('Training end!')

        self.save_final_model()
        self.conduct_final_validation()
