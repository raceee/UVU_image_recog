import os
from skimage import io
import torch
from torch.utils.data import Dataset
import pandas as pd


class MetalicDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.data.iloc[index, 0])
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.data.iloc[index, 1]))
        if self.transform:
            image = self.transform(image)
            if image.shape[0] < 3:
                fill = 3 - image.shape[0]
                pad = torch.zeros(fill, 244, 244)
                image = torch.cat([image, pad], dim=0)
            elif image.shape[0] > 3:
                image = image[:3, :, :]
        return (image, y_label)

