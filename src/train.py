import argparse
import os
import boto3
import zipfile
import tarfile
import torch
import torch.distributed as dist
from collections import Counter
import torch.nn as nn
import torch.optim as optim
from transformers import VivitForVideoClassification
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from dataset import ViolentVideoDataset
from torch.amp import GradScaler, autocast
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tqdm import tqdm

# TODO: frame size = 244* 244

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
        num_frames=10,
        ignore_mismatched_sizes=True,
        use_safetensors=True,
    )
    model.config.id2label = {0: "non_violent", 1: "violent"}
    model.config.label2id = {"non_violent": 0, "violent": 1}
    for param in model.vivit.parameters():
        param.requires_grad = False

    model = model.to(device)
    if is_dist: model = DDP(model, device_ids=[local_rank])
    return model

def download_data(bucket: str, key: str, local_path: str, data_dir: str):
    s3 = boto3.client('s3')
    response = s3.download_file(Bucket=bucket, Key=key, Filename=local_path)
    print("Downloaded data!")
    with zipfile.ZipFile(local_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    print("Data extracted")

def sliding_windows(video, window_size=10, stride=5) -> torch.Tensor:
    T = video.shape[0]
    windows = []

    for start in range(0, T-window_size+1, stride):
        windows.append(video[start:start+window_size])

    if len(windows) == 0:
        if T < window_size:
            pad = video[-1:].repeat(window_size - T, 1, 1, 1)
            video = torch.cat((video, pad), 0)
        windows.append(video)
    return torch.stack(windows)


def train(
        epochs: int,
        model,
        train_loader,
        val_loader,
        train_sampler,
        optimizer,
        scheduler,
        criterion,
        device,
        is_dist,
        checkpoint_dir,
        window_size=10,
        stride=5,
        backbone_unfreeze_epoch=5
):
    print("Training Vivit Model..")
    base_model = model.module if is_dist else model
    base_model.gradient_checkpointing_enable()
    scaler = GradScaler()
    best_val_loss = float('inf')

    for epoch in range(epochs):
        if epoch == backbone_unfreeze_epoch:
            base = model.module if is_dist else model
            print("Unfreezed Vivit backbone")
            for param in base.vivit.parameters():
                param.requires_grad = True
        if is_dist and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        model.train()
        running_loss = 0.0

        for batch_idx, (videos, labels) in enumerate(train_loader):
            labels = labels.to(device)
            optimizer.zero_grad()
            batch_video_logits = []
            for video in videos:
                video = video.to(device)
                if video.shape[0] == 3:
                    video = video.permute(1, 0, 2, 3)
                windows = sliding_windows(video, window_size=window_size, stride=stride).to(device)

                with autocast(device_type="cuda"):
                    outputs = model(windows).logits
                    video_logits = torch.logsumexp(outputs, dim=0)

                batch_video_logits.append(video_logits)
            batch_video_logits = torch.stack(batch_video_logits)

            with autocast(device_type="cuda"):
                loss  = criterion(batch_video_logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        scheduler.step()
        avg_train_loss = running_loss / len(train_loader)
        print(f"--- Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} ---")

        # Run validation
        val_loss, val_acc = validate(model, val_loader, criterion, device, is_dist, window_size=window_size, stride=stride)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': (model.module if is_dist else model).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, os.path.join(checkpoint_dir, "best_model.pt"))
            print(f"Best model saved (val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%)")


def validate(model, val_loader, criterion, device, is_dist, window_size=10, stride=5):
    model.eval()
    running_loss=0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (videos, labels) in enumerate(val_loader):
            labels = labels.to(device)
            batch_video_logits = []

            for video in videos:
                video = video.to(device)
                if video.shape[0] == 3:
                    video = video.permute(1, 0, 2, 3)
                windows = sliding_windows(video, window_size=window_size, stride=stride).to(device)

                with autocast(device_type="cuda"):
                    outputs = model(windows).logits
                    video_logits = torch.logsumexp(outputs, dim=0)

                batch_video_logits.append(video_logits)
                torch.cuda.empty_cache()

            batch_video_logits = torch.stack(batch_video_logits)
            loss = criterion(batch_video_logits, labels)
            running_loss += loss.item()

            preds = torch.argmax(batch_video_logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss/len(val_loader)
    accuracy = correct / total * 100
    print(f"    Val Loss: {avg_loss:.4f} | Val Accuracy: {accuracy:.2f}% ({correct}/{total})")
    return avg_loss, accuracy

def test_model(model, test_loader, device, window_size=10, stride=5):
    model.eval()

    all_preds = []
    all_labels = []

    pbar = tqdm(test_loader, desc="Testing", dynamic_ncols=True)

    with torch.no_grad():
        for videos, labels in pbar:
            labels = labels.to(device)

            batch_video_logits = []

            for video in videos:
                video = video.to(device)
                # ensure shape [T, C, H, W]
                if video.shape[0] == 3:
                    video = video.permute(1, 0, 2, 3)

                # sliding windows
                windows = sliding_windows(
                    video,
                    window_size=window_size,
                    stride=stride
                ).to(device)

                # forward pass
                outputs = model(windows).logits  # [num_windows, num_classes]

                # MIL aggregation (same as validation)
                video_logits = torch.logsumexp(outputs, dim=0)

                batch_video_logits.append(video_logits)

            batch_video_logits = torch.stack(batch_video_logits)

            preds = torch.argmax(batch_video_logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # ===== METRICS =====
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=["non_violent", "violent"])

    print("\n===== FINAL TEST RESULTS =====")
    print(f"Accuracy: {acc * 100:.2f}%\n")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    return acc, cm, report

def collate(batch):
    videos, labels = zip(*batch)
    return list(videos), torch.stack(labels)

def get_dataloader(datapath: str, is_dist: bool, augment=True, num_workers = 2, batch_size = 16):
    dataset = ViolentVideoDataset(datapath, augment)
    distributed_sampler = None
    shuffle_data = True

    if is_dist:
        distributed_sampler = DistributedSampler(dataset, shuffle=True)
        shuffle_data = False
    num_workers = 2 if is_dist else 0
    # TODO: Data must be pulled at random from dataset
    dataloader =  DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = shuffle_data,
        sampler = distributed_sampler,
        num_workers = num_workers,
        collate_fn = collate
    )
    return dataloader, distributed_sampler

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoint')
    parser.add_argument('--model-dir', type=str, default='./model')
    parser.add_argument('--data-dir', type=str, default='./data')
    return parser.parse_args()


if __name__ == '__main__':
    is_dist, local_rank = init_ddp()
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = get_device(local_rank, is_dist)

    model = get_model(device, is_dist, local_rank)
    #
    train_data_path = os.path.join(args.data_dir, 'train')
    test_data_path = os.path.join(args.data_dir, 'test')
    valid_data_path = os.path.join(args.data_dir, 'valid')
    #
    bucket = 'agression-model'
    file_name = 'data/videos'
    download_data(bucket, file_name, f"{args.data_dir}/videos", args.data_dir)
    #
    train_loader, train_sampler = get_dataloader(train_data_path, is_dist, batch_size=args.batch_size)
    valid_loader, _ = get_dataloader(valid_data_path, is_dist, batch_size=args.batch_size, augment=False)
    test_loader, _= get_dataloader(test_data_path, is_dist, batch_size=args.batch_size, augment=False)
    labels = train_loader.dataset.labels
    frame_counts = []
    for path in train_loader.dataset.tensor_paths:
        t = torch.load(path, weights_only=True)
        frame_counts.append(t.shape[1])  # shape is [C, T, H, W] so dim 1 is T

    print(f"Min frames: {min(frame_counts)}")
    print(f"Max frames: {max(frame_counts)}")
    print(f"Mean frames: {sum(frame_counts) / len(frame_counts):.1f}")
    print(f"Videos shorter than 10: {sum(1 for f in frame_counts if f < 10)}")
    print(f"Videos shorter than 32: {sum(1 for f in frame_counts if f < 32)}")
    n_nonviolent = sum(1 for l in labels if l == 0)
    n_violent = sum(1 for l in labels if l == 1)
    total = n_nonviolent + n_violent

    weights = torch.tensor([
        total / (2 * n_nonviolent),
        total / (2 * n_violent),
    ], dtype=torch.float).to(device)

    print(f"Class weights: non_violent={weights[0]:.3f}, violent={weights[1]:.3f}")

    optimizer = optim.AdamW([
        {'params': (model.module if is_dist else model).vivit.parameters(), 'lr': 1e-5},
        {'params': (model.module if is_dist else model).classifier.parameters(), 'lr': 1e-4}
    ])
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    train(args.epochs, model, train_loader, valid_loader, train_sampler, optimizer, scheduler, criterion, device, is_dist, checkpoint_dir=args.checkpoint_dir)

   #  s3 = boto3.client("s3")
   #  local_p = "./model/model.pt"
   #  bucket = "agression-model"
   #  key = "vivit/checkpoints/best_model.pt"
   # # s3.download_file(bucket, key, local_p)
   #  checkpoint = torch.load(local_p, map_location=device)
   #  model.load_state_dict(checkpoint['model_state_dict'])
    test_model(model, test_loader, device)


