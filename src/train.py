import argparse
import os
import boto3
import zipfile
import tarfile
from torch.utils.data import DataLoader, DistributedSampler

from src.dataset import ViolentVideoDataset


def download_data(bucket: str, key: str, local_path: str):
    # s3 = boto3.client('s3')
    # response = s3.download_file(Bucket=bucket, Key=key, Filename=local_path)
    # print("Downloaded data!")
    with zipfile.ZipFile(local_path, 'r') as zip_ref:
        zip_ref.extractall("../data")
    print("Data extracted")

def train(epochs: int):
    pass


def validate():
    pass


def test():
    pass


def get_dataloader(datapath: str):
    dataset = ViolentVideoDataset(datapath)
    distributed_sampler = None
    # TODO: Data must be pulled at random from dataset
    dataloader = None
    return

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--checkpoint_dir', type=str, default='../checkpoint')
    parser.add_argument('--model-dir', type=str, default='../model')
    parser.add_argument('--data_dir', type=str, default='../data')
    return parser.parse_args()



if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train_data_path = os.path.join(args.data_dir, 'train')
    test_data_path = os.path.join(args.data_dir, 'test')
    valid_data_path = os.path.join(args.data_dir, 'val')

    bucket = 'agression-model'
    file_name = 'data/videos'
    download_data(bucket, file_name, f"{args.data_dir}/videos")

    # train_loader = get_dataloader(args.data_dir)
    # valid_loader = get_dataloader(args.data_dir)
    # test_loader = get_dataloader(args.data_dir)



