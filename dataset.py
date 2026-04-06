import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class LungDataset(Dataset):
    def __init__(self, image_dir, mask_dir, size=(256, 256)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.files = sorted(os.listdir(image_dir))
        self.size = size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]

        image_path = os.path.join(self.image_dir, fname)
        mask_path = os.path.join(self.mask_dir, fname)

        # Load
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Resize
        image = cv2.resize(image, self.size)
        mask = cv2.resize(mask, self.size)

        # Normalize
        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0

        # Binary mask
        mask = (mask > 0.5).astype(np.float32)

        # Add channel dimension
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        # Convert to tensor
        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        return image, mask