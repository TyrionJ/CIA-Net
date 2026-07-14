import argparse

from scripts.preprocessor import Processor


def run_preprocess():
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', type=str, default='../sample_set/CIA-Net_processed', help='processed folder')
    parser.add_argument('-r', type=str, default='../sample_set/CIA-Net_raw', help='raw folder')
    parser.add_argument('-D', type=int, default=1)
    args = parser.parse_args()

    Processor(raw_dir=args.r, processed_dir=args.p, dataset_id=args.D).run()


if __name__ == '__main__':
    run_preprocess()
