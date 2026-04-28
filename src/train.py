import argparse
import os
import boto3
import zipfile
import tarfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from transformers import VivitForVideoClassification
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from dataset import ViolentVideoDataset

# init DDP
def init_ddp():
    if "RANK" in os.environ:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return True, local_rank
    return False, None


def get_device(local_rank, use_ddp):
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}" if use_ddp else "cuda:0")
    return torch.device("cpu")
 


def get_model(device, is_dist, local_rank):
    model = VivitForVideoClassification.from_pretrained(
        "google/vivit-b-16x2-kinetics400", 
        num_labels=2, 
        ignore_mismatched_sizes=True
    )
    model = model.to(device)
    if is_dist: model = DDP(model, device_ids=[local_rank])
    return model

def download_data(bucket: str, key: str, local_path: str):
    s3 = boto3.client('s3')
    response = s3.download_file(Bucket=bucket, Key=key, Filename=local_path)
    print("Downloaded data!")
    with zipfile.ZipFile(local_path, 'r') as zip_ref:
        zip_ref.extractall("./data")
    print("Data extracted")

def train(epochs: int, model, train_loader, train_sampler, optimizer, criterion, device, is_dist):
    print("Training Began...")

    for epoch in range(epochs):
        if is_dist and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        model.train()
        running_loss = 0.0

        for batch_idx, (videos, labels) in enumerate(train_loader):
            videos, labels = videos.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(videos).logits
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Batch {batch_idx} | Loss: {loss.item():.4f}")
        print(f"--- Epoch {epoch+1} Complete. Avg Loss: {running_loss/len(train_loader):.4f} ---")


def validate():
    pass


def test():
    pass


def get_dataloader(datapath: str, is_dist: bool, num_workers = 2, augment = False, batch_size = 16):
    dataset = ViolentVideoDataset(datapath)
    distributed_sampler = None
    shuffle_data = True

    if is_dist:
        distributed_sampler = DistributedSampler(dataset, shuffle=True)
        shuffle_data = False

    # TODO: Data must be pulled at random from dataset
    dataloader =  DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = shuffle_data,
        sampler = distributed_sampler,
        num_workers = num_workers
    )
    return dataloader, distributed_sampler

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint')
    parser.add_argument('--model-dir', type=str, default='./model')
    parser.add_argument('--data_dir', type=str, default='./data')
    return parser.parse_args()



if __name__ == '__main__':
    is_dist, local_rank = init_ddp()
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = get_device(local_rank, is_dist)

    model = get_model(device, is_dist, local_rank)

    train_data_path = os.path.join(args.data_dir, 'train')
    test_data_path = os.path.join(args.data_dir, 'test')
    valid_data_path = os.path.join(args.data_dir, 'val')

    bucket = 'agression-model'
    file_name = 'data/videos'
    #download_data(bucket, file_name, f"{args.data_dir}/videos")

    train_loader, train_sampler = get_dataloader(train_data_path, is_dist, batch_size=args.batch_size)
    # valid_loader = get_dataloader(args.data_dir)
    # test_loader = get_dataloader(args.data_dir)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    train(args.epochs, model, train_loader, train_sampler, optimizer, criterion, device, is_dist)