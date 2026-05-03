import torch
from torch.utils.data import Dataset
import os
from torchvision import transforms as T
import random

spatial_T = T.Compose([
            T.RandomResizedCrop((224, 224)),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.05),
        ])

def temporal_T(video):
    num_frames = video.shape[0]
    mask_size = 2

    if random.random() < 0.5 and num_frames > 5:
        shift = random.randint(-2, 2)
        video = torch.roll(video, shifts=shift, dims=0)

    # Change speed. Remove one in every x frames
    if random.random() < 0.3 and num_frames > 10:
        stride = random.choice([2, 3])
        video = video[::stride]

    if random.random() < 0.3:
        t0 = random.randint(0, num_frames - mask_size)
        video[t0:t0 + mask_size] = 0

    return video

def transforms(video):
    video = temporal_T(video)
    seed = random.randint(0, 1_000_000)

    out = []
    for frame in video:
        random.seed(seed)
        torch.manual_seed(seed)
        out.append(spatial_T(frame))
    return torch.stack(out)


class ViolentVideoDataset(Dataset):
    def __init__(self, data_dir, augment=True):
        self.tensor_paths: list[str] = []   # file names Same index
        self.labels: list[int] = []
        self.spatial_T = T.Compose([
            T.RandomResizedCrop((224, 224)),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.05),
        ])
        self.transforms = transforms
        self.augment = augment

        video_dir = os.path.join(data_dir, 'video')
        labels_dir = os.path.join(data_dir, 'labels')

        for each in os.listdir(video_dir):
            if each.endswith('.pt'):
                self.tensor_paths.append(os.path.join(video_dir, each))
                label_file = each.replace('.pt', '.txt')
                label_path = os.path.join(labels_dir, label_file)
                with open(label_path, 'r') as f:
                    self.labels.append(int(f.read().strip()))

    def __len__(self):
        return len(self.tensor_paths)

    def __getitem__(self, idx):
        tensor_path = self.tensor_paths[idx]
        label = self.labels[idx]

        video = torch.load(tensor_path, weights_only=True).float() / 255.0
        video = video.permute(1, 0, 2, 3)
        label = torch.tensor(label, dtype=torch.long)

        if self.augment:
            video = self.transforms(video)

        return video, label
