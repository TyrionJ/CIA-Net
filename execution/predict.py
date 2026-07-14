import argparse

from network import *
from scripts.predictor import Predictor


def run_predict():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', type=str, default='../sample_set/CIA-Net_raw/Dataset001_SAMPLE/imagesTs;masksTs', help='Input folder with images to predict (.nii.gz)')
    parser.add_argument('-o', type=str, default='../sample_set/CIA-Net_raw/Dataset001_SAMPLE/preds', help='Prediction to store')
    parser.add_argument('-r', type=str, default='../sample_set/CIA-Net_results', help='Task results folder')
    parser.add_argument('-f', type=str, default='all', help='Used folds')
    parser.add_argument('-d', type=str, default='0', help='device: cpu or 0, 1, 2, ...')
    parser.add_argument('-D', type=int, default=1, help='dataset ID')
    parser.add_argument('-N', type=str, default='CIANet', choices=support_networks, help='Task network')
    parser.add_argument('-t', type=str, default='Task01_CIA-Net', help='Task name for the dataset')
    parser.add_argument('-p', type=int, default=1, help='Using post process')
    parser.add_argument('--c', action='store_true', help='Clear predicted')
    args = parser.parse_args()
    args.p = args.p > 0

    Predictor(results_dir=args.r, dataset_id=args.D, task_name=args.t, net_class=eval(args.N), folds=args.f,
              input_dirs=args.i.split(';'), output_dir=args.o, device=args.d, clear=args.c).run(args.p)


if __name__ == '__main__':
    run_predict()
