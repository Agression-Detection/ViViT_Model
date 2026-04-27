from uuid import SafeUUID

import torch
from torch.utils.data import Dataset
import os

class ViolentVideoDataset(Dataset):
    def __init__(self, tensor_dir):
        self.tensor_paths = []
        self.labels = []

        for label_name, label_val in [("Safe", 0), ("Violent", 1)]:
            folder_path = os.path.join(tensor_dir, label_name)
            if os.path.exists(folder_path):
                for file in os.listdir(folder_path):
                    if file.endswith('.pt'):
                        self.tensor_paths.append(os.path.join(folder_path, file))
                        self.labels.append(label_val)

    def __len__(self):
        return len(self.tensor_paths)

    def __getitem__(self, idx):
        tensor_path = self.tensor_paths[idx]
        label = self.labels[idx]

        video = torch.load(tensor_path).float() / 255.0
        label = torch.tensor(label, dtype=torch.long)

        return video, label
