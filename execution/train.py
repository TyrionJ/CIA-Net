import torch
import argparse

from network import *
from net_loss import *
from scripts.trainer import NetTrainer

torch.autograd.set_detect_anomaly(True)


def run_trainer():
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', type=str, default='../sample_set/CIA-Net_processed', help='processed folder')
    parser.add_argument('-r', type=str, default='../sample_set/CIA-Net_results', help='results folder')
    parser.add_argument('-f', type=int, default=0, help='fold')
    parser.add_argument('-D', type=int, default=1, help='dataset ID')
    parser.add_argument('-t', type=str, default='Task01_CIA-Net', help='Task name for the dataset')
    parser.add_argument('-N', type=str, default='CIANet', choices = support_networks, help='Task network')
    parser.add_argument('-L', type=str, default='CIANetLoss', choices = support_loss_fns, help='Train loss function')
    parser.add_argument('-ps', type=str, default='[128, 128, 80]', help='Train patch size')

    parser.add_argument('-d', type=str, default='2', help='device: cpu or 0, 1, 2, ...')
    parser.add_argument('-e', type=int, default=150, help='epoch number')
    parser.add_argument('-b', type=int, default=2, help='batch size')
    parser.add_argument('--c', action='store_true', help='continue train')
    parser.add_argument('--v', action='store_true', help='only validation if train finished')
    args = parser.parse_args()

    tr = NetTrainer(processed_dir=args.p,
                    result_dir=args.r,
                    dataset_id=args.D,
                    task_name=args.t,
                    batch_size=args.b,
                    patch_size=eval(args.ps),
                    net_class=eval(args.N),
                    loss_fn=eval(args.L),
                    fold=args.f,
                    go_on=args.c,
                    epochs=args.e,
                    device=args.d,
                    validation=args.v,
                    logger=print)
    tr.run()


if __name__ == '__main__':
    run_trainer()
