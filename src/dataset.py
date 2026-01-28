import torch
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class MoonDataset(Dataset):
    def __init__(self, manifest_path, split='train', transform=None):
        df = pd.read_csv(manifest_path)
        self.data = df[df['split'] == split].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        image = cv2.imread(row['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(row['mask_path'], cv2.IMREAD_COLOR)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB) 
        mask = self._preprocess_mask(mask)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask.long() # Model SMP wymaga maski typu Long 

    def _preprocess_mask(self, mask):
        new_mask = np.zeros(mask.shape[:2], dtype=np.longlong)
        
        new_mask[(mask[..., 2] > 128)] = 1 
        new_mask[(mask[..., 0] > 128)] = 2 
        new_mask[(mask[..., 1] > 128)] = 3 
        
        return new_mask

def get_training_augmentation():
    return A.Compose([
        A.Resize(256, 256), 
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.GaussNoise(p=0.2),
            A.RandomBrightnessContrast(p=0.5),
        ], p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_validation_augmentation():
    return A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])