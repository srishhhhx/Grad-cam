import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import warnings
from typing import Dict, List, Tuple, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PsoriasisDataset(Dataset):
    """
    Dataset class for psoriasis segmentation using COCO format annotations
    """
    def __init__(self, root_dir, split='train', transform=None, image_size=512):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.image_size = image_size
        
        # Initialize COCO API
        ann_file = os.path.join(root_dir, split, '_annotations.coco.json')
        self.coco = COCO(ann_file)
        
        # Get all image IDs
        self.image_ids = list(self.coco.imgs.keys())
        
        print(f"Loaded {len(self.image_ids)} images for {split} split")
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        # Get image info
        image_id = self.image_ids[idx]
        image_info = self.coco.imgs[image_id]
        
        # Load image
        image_path = os.path.join(self.root_dir, self.split, image_info['file_name'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get annotations for this image
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # Create mask
        mask = np.zeros((image_info['height'], image_info['width']), dtype=np.uint8)
        
        for ann in anns:
            # Convert COCO segmentation to mask
            if 'segmentation' in ann:
                if isinstance(ann['segmentation'], list):
                    # Polygon format
                    rles = maskUtils.frPyObjects(ann['segmentation'], 
                                               image_info['height'], 
                                               image_info['width'])
                    rle = maskUtils.merge(rles)
                elif isinstance(ann['segmentation'], dict):
                    # RLE format
                    rle = ann['segmentation']
                else:
                    continue
                    
                ann_mask = maskUtils.decode(rle)
                mask = np.maximum(mask, ann_mask)
        
        # Resize image and mask (preserve aspect ratio for 960x960)
        if image.shape[:2] != (self.image_size, self.image_size):
            # For 960x960 images, resize appropriately
            image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        
        # Apply transformations
        if self.transform:
            # Ensure numpy arrays for albumentations
            if isinstance(image, torch.Tensor):
                image = image.numpy()
            if isinstance(mask, torch.Tensor):
                mask = mask.numpy()
            
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Convert to tensors
        if not torch.is_tensor(image):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        if not torch.is_tensor(mask):
            mask = torch.from_numpy(mask).float()
            # Ensure mask has correct dimensions [1, H, W]
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)
        
        return {
            'image': image,
            'mask': mask,
            'image_id': image_id,
            'filename': image_info['file_name']
        }

def get_transforms(image_size=960, is_training=True):
    """
    Get augmentation transforms for training and validation (optimized for 960x960)
    """
    if is_training:
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                border_mode=cv2.BORDER_REFLECT_101,  # Reflect edges
                p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
            ], p=0.3),
            A.OneOf([
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=20,
                    val_shift_limit=10,
                    p=1.0
                ),
                A.CLAHE(clip_limit=2.0, p=1.0),
            ], p=0.3),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=1.0
            ),
            ToTensorV2()
        ])
    else:
        transform = A.Compose([
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=1.0
            ),
            ToTensorV2()
        ])
    
    return transform

def get_loaders(root_dir, batch_size=4, num_workers=4, image_size=960):
    """
    Create data loaders for train, validation, and test sets
    """
    # Define transforms
    train_transform = get_transforms(image_size=image_size, is_training=True)
    val_transform = get_transforms(image_size=image_size, is_training=False)
    
    # Create datasets
    train_dataset = PsoriasisDataset(
        root_dir=root_dir,
        split='train',
        transform=train_transform,
        image_size=image_size
    )
    
    valid_dataset = PsoriasisDataset(
        root_dir=root_dir,
        split='valid',
        transform=val_transform,
        image_size=image_size
    )
    
    test_dataset = PsoriasisDataset(
        root_dir=root_dir,
        split='test',
        transform=val_transform,
        image_size=image_size
    )
    
    # Create data loaders
    loaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        ),
        'valid': DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        ),
        'test': DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    }
    
    print(f"Created loaders - Train: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}")
    
    return loaders 