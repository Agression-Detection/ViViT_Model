import torch
from torch.utils.data import Dataset
import os

class ViolentVideoDataset(Dataset):
    def __init__(self, data_dir):
        self.tensor_paths: list[str] = []   # file names Same index
        self.labels: list[int] = []         # values in labels

        video_dir = os.path.join(data_dir, 'video')
        labels_dir = os.path.join(data_dir, 'labels')

        for each in os.listdir(video_dir):
            if each.endswith('.pt'):
                self.tensor_paths.append(os.path.join(video_dir, each))
                label_file = each.replace('.pt', '.txt')
                label_path = os.path.join(labels_dir, label_file)
                with open(label_path, 'r') as f:
                    self.labels.append(int(f.read().strip()))

        print(self.tensor_paths[:5])
        print(self.labels[:5])

    def __len__(self):
        return len(self.tensor_paths)

    def __getitem__(self, idx):
        tensor_path = self.tensor_paths[idx]
        label = self.labels[idx]

        video = torch.load(tensor_path, weights_only=True).float() / 255.0
        label = torch.tensor(label, dtype=torch.long)

        return video, label
