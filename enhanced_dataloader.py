"""
Enhanced Data Loader for U-Net Segmentation
Specifically designed for Dataset_no_preprocessing with COCO format annotations

Features:
- Comprehensive data augmentation pipeline
- Advanced preprocessing techniques
- Robust error handling and validation
- Memory-efficient loading
- Support for different input sizes
- Class balancing and weighted sampling
"""

import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import warnings
from typing import Dict, List, Tuple, Optional, Union
import logging
from collections import Counter
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'


class EnhancedSegmentationDataset(Dataset):
    """
    Enhanced Dataset class for segmentation using COCO format annotations
    
    Features:
    - Advanced preprocessing and augmentation
    - Memory-efficient loading
    - Robust error handling
    - Support for class balancing
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[A.Compose] = None,
        image_size: int = 640,
        cache_images: bool = False,
        validate_data: bool = True,
        min_mask_area: int = 100,
        include_negative_samples: bool = True,
        handle_missing_annotations: str = 'warn',
        augmentation_multiplier: int = 1,
        deterministic_augmentation: bool = False
    ):
        """
        Initialize the dataset

        Args:
            root_dir: Path to dataset root directory
            split: Dataset split ('train', 'valid', 'test')
            transform: Albumentations transform pipeline
            image_size: Target image size for resizing
            cache_images: Whether to cache images in memory (for small datasets)
            validate_data: Whether to validate data integrity
            min_mask_area: Minimum mask area to consider valid
            include_negative_samples: Whether to include images without annotations as negative samples
            handle_missing_annotations: How to handle missing annotations ('warn', 'skip', 'silent')
            augmentation_multiplier: Number of augmented versions per original image (1=no extra, 3=1:3 ratio)
            deterministic_augmentation: Whether to use deterministic augmentation (same seed for reproducibility)
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.image_size = image_size
        self.cache_images = cache_images
        self.min_mask_area = min_mask_area
        self.include_negative_samples = include_negative_samples
        self.handle_missing_annotations = handle_missing_annotations
        self.augmentation_multiplier = augmentation_multiplier
        self.deterministic_augmentation = deterministic_augmentation
        
        # Initialize COCO API
        ann_file = os.path.join(root_dir, split, '_annotations.coco.json')
        if not os.path.exists(ann_file):
            raise FileNotFoundError(f"Annotation file not found: {ann_file}")
            
        self.coco = COCO(ann_file)
        
        # Get all image IDs and filter valid ones
        self.image_ids = list(self.coco.imgs.keys())
        
        if validate_data:
            self.image_ids = self._validate_dataset()
        
        # Cache for images if enabled
        self.image_cache = {} if cache_images else None
        
        # Calculate dataset statistics
        self.stats = self._calculate_stats()
        
        logger.info(f"Loaded {len(self.image_ids)} valid images for {split} split")
        logger.info(f"Dataset stats: {self.stats}")
    
    def _validate_dataset(self) -> List[int]:
        """Validate dataset and filter out corrupted/invalid samples"""
        valid_ids = []
        
        for image_id in self.image_ids:
            try:
                image_info = self.coco.imgs[image_id]
                image_path = os.path.join(self.root_dir, self.split, image_info['file_name'])
                
                # Check if image file exists
                if not os.path.exists(image_path):
                    logger.warning(f"Image file not found: {image_path}")
                    continue
                
                # Check if image can be loaded
                img = cv2.imread(image_path)
                if img is None:
                    logger.warning(f"Cannot load image: {image_path}")
                    continue
                
                # Check annotations
                ann_ids = self.coco.getAnnIds(imgIds=image_id)
                if len(ann_ids) == 0:
                    if not self.include_negative_samples:
                        if self.handle_missing_annotations == 'warn':
                            logger.warning(f"Skipping image without annotations: {image_info['file_name']}")
                        continue
                    else:
                        # Include as negative sample
                        if self.handle_missing_annotations == 'warn':
                            logger.info(f"Including image without annotations as negative sample: {image_info['file_name']}")
                        elif self.handle_missing_annotations == 'skip':
                            continue
                
                valid_ids.append(image_id)
                
            except Exception as e:
                logger.warning(f"Error validating image {image_id}: {str(e)}")
                continue
        
        logger.info(f"Validated {len(valid_ids)}/{len(self.image_ids)} images")
        return valid_ids
    
    def _calculate_stats(self) -> Dict:
        """Calculate dataset statistics"""
        total_pixels = 0
        positive_pixels = 0
        mask_areas = []
        
        # Sample a subset for statistics (to avoid loading all images)
        sample_size = min(100, len(self.image_ids))
        sample_ids = np.random.choice(self.image_ids, sample_size, replace=False)
        
        for image_id in sample_ids:
            try:
                mask = self._load_mask(image_id)
                total_pixels += mask.size
                positive_pixels += np.sum(mask > 0)
                mask_areas.append(np.sum(mask > 0))
            except:
                continue
        
        pos_ratio = positive_pixels / total_pixels if total_pixels > 0 else 0
        
        return {
            'positive_ratio': pos_ratio,
            'negative_ratio': 1 - pos_ratio,
            'mean_mask_area': np.mean(mask_areas) if mask_areas else 0,
            'std_mask_area': np.std(mask_areas) if mask_areas else 0,
            'total_samples': len(self.image_ids)
        }
    
    def _load_image(self, image_id: int) -> np.ndarray:
        """Load and preprocess image"""
        if self.image_cache and image_id in self.image_cache:
            return self.image_cache[image_id].copy()
        
        image_info = self.coco.imgs[image_id]
        image_path = os.path.join(self.root_dir, self.split, image_info['file_name'])
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize if needed
        if image.shape[:2] != (self.image_size, self.image_size):
            image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        
        # Cache if enabled
        if self.image_cache is not None:
            self.image_cache[image_id] = image.copy()
        
        return image
    
    def _load_mask(self, image_id: int) -> np.ndarray:
        """Load and create segmentation mask"""
        image_info = self.coco.imgs[image_id]
        
        # Get annotations for this image
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # Create mask
        mask = np.zeros((image_info['height'], image_info['width']), dtype=np.uint8)
        
        for ann in anns:
            if 'segmentation' in ann:
                try:
                    if isinstance(ann['segmentation'], list):
                        # Polygon format
                        rles = maskUtils.frPyObjects(
                            ann['segmentation'], 
                            image_info['height'], 
                            image_info['width']
                        )
                        rle = maskUtils.merge(rles)
                    elif isinstance(ann['segmentation'], dict):
                        # RLE format
                        rle = ann['segmentation']
                    else:
                        continue
                    
                    ann_mask = maskUtils.decode(rle)
                    mask = np.maximum(mask, ann_mask)
                    
                except Exception as e:
                    logger.warning(f"Error processing annotation: {str(e)}")
                    continue
        
        # Resize mask if needed
        if mask.shape != (self.image_size, self.image_size):
            mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        
        return mask
    
    def __len__(self) -> int:
        # Return original length multiplied by augmentation factor
        return len(self.image_ids) * (1 + self.augmentation_multiplier)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample with optional augmentation multiplier"""
        try:
            # Calculate which original image and which augmentation version
            original_dataset_size = len(self.image_ids)
            original_idx = idx % original_dataset_size
            augmentation_idx = idx // original_dataset_size

            image_id = self.image_ids[original_idx]

            # Load image and mask
            image = self._load_image(image_id)
            mask = self._load_mask(image_id)
            
            # Apply transformations with deterministic seeding if needed
            if self.transform:
                if self.deterministic_augmentation and augmentation_idx > 0:
                    # Set deterministic seed based on image_id and augmentation_idx
                    seed = hash((image_id, augmentation_idx)) % (2**32)
                    np.random.seed(seed)
                    torch.manual_seed(seed)

                transformed = self.transform(image=image, mask=mask)
                image = transformed['image']
                mask = transformed['mask']
            
            # Convert to tensors if not already
            if not torch.is_tensor(image):
                image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            if not torch.is_tensor(mask):
                mask = torch.from_numpy(mask).float()
                if len(mask.shape) == 2:
                    mask = mask.unsqueeze(0)
            
            # Get additional info
            image_info = self.coco.imgs[image_id]
            
            return {
                'image': image,
                'mask': mask,
                'image_id': image_id,
                'filename': f"{image_info['file_name']}_aug{augmentation_idx}",
                'original_filename': image_info['file_name'],
                'original_size': (image_info['height'], image_info['width']),
                'augmentation_idx': augmentation_idx,
                'is_augmented': augmentation_idx > 0
            }
            
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {str(e)}")
            # Return a dummy sample to avoid breaking the training loop
            return {
                'image': torch.zeros(3, self.image_size, self.image_size),
                'mask': torch.zeros(1, self.image_size, self.image_size),
                'image_id': -1,
                'filename': 'error_aug0',
                'original_filename': 'error',
                'original_size': (self.image_size, self.image_size),
                'augmentation_idx': 0,
                'is_augmented': False
            }
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for balanced training"""
        pos_ratio = self.stats['positive_ratio']
        neg_ratio = self.stats['negative_ratio']
        
        # Calculate weights inversely proportional to class frequency
        pos_weight = 1.0 / (pos_ratio + 1e-6)
        neg_weight = 1.0 / (neg_ratio + 1e-6)
        
        # Normalize weights
        total_weight = pos_weight + neg_weight
        pos_weight /= total_weight
        neg_weight /= total_weight
        
        return torch.tensor([neg_weight, pos_weight])


def get_enhanced_transforms(image_size: int = 640, is_training: bool = True, augmentation_level: str = 'medium') -> A.Compose:
    """
    Get comprehensive augmentation transforms

    Args:
        image_size: Target image size
        is_training: Whether this is for training (applies augmentations)
        augmentation_level: 'light', 'medium', 'heavy' - controls augmentation intensity
    """

    if not is_training:
        # Validation/test transforms - only normalization
        return A.Compose([
            A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ])

    # Training transforms with different intensity levels
    base_transforms = [
        A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
    ]

    if augmentation_level == 'light':
        augmentations = [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.3),
        ]

    elif augmentation_level == 'medium':
        augmentations = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.1, rotate_limit=15,
                border_mode=cv2.BORDER_REFLECT_101, p=0.5
            ),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
                A.MotionBlur(blur_limit=3, p=1.0),
            ], p=0.3),
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=1.0),
                A.CLAHE(clip_limit=2.0, p=1.0),
                A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            ], p=0.3),
        ]

    elif augmentation_level == 'heavy':
        augmentations = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.2),
            A.ShiftScaleRotate(
                shift_limit=0.15, scale_limit=0.15, rotate_limit=25,
                border_mode=cv2.BORDER_REFLECT_101, p=0.7
            ),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 80.0), p=1.0),
                A.GaussianBlur(blur_limit=5, p=1.0),
                A.MotionBlur(blur_limit=5, p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
            ], p=0.4),
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=15, p=1.0),
                A.CLAHE(clip_limit=3.0, p=1.0),
                A.RandomGamma(gamma_limit=(70, 130), p=1.0),
                A.ChannelShuffle(p=1.0),
            ], p=0.4),
            A.CoarseDropout(max_holes=6, max_height=24, max_width=24, p=0.2),
            A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.2),
            A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=0.2),
        ]

    else:
        raise ValueError(f"Unknown augmentation level: {augmentation_level}")

    # Final transforms
    final_transforms = [
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ]

    return A.Compose(base_transforms + augmentations + final_transforms)


def create_weighted_sampler(dataset: EnhancedSegmentationDataset) -> WeightedRandomSampler:
    """Create weighted sampler for balanced training"""
    weights = []

    for idx in range(len(dataset)):
        try:
            sample = dataset[idx]
            mask = sample['mask']

            # Calculate positive ratio for this sample
            pos_pixels = torch.sum(mask > 0).item()
            total_pixels = mask.numel()
            pos_ratio = pos_pixels / total_pixels

            # Assign higher weight to samples with more balanced masks
            if pos_ratio < 0.01:  # Very few positive pixels
                weight = 0.5
            elif pos_ratio > 0.5:  # Mostly positive pixels
                weight = 0.8
            else:  # Balanced samples
                weight = 1.0

            weights.append(weight)

        except:
            weights.append(0.1)  # Low weight for problematic samples

    return WeightedRandomSampler(weights, len(weights), replacement=True)


def get_enhanced_loaders(
    root_dir: str,
    batch_size: int = 4,
    num_workers: int = 4,
    image_size: int = 640,
    augmentation_level: str = 'medium',
    use_weighted_sampling: bool = False,
    cache_images: bool = False,
    validate_data: bool = True,
    include_negative_samples: bool = True,
    handle_missing_annotations: str = 'warn',
    augmentation_multiplier: int = 0,
    deterministic_augmentation: bool = False
) -> Dict[str, DataLoader]:
    """
    Create enhanced data loaders for train, validation, and test sets

    Args:
        root_dir: Path to dataset directory
        batch_size: Batch size for training
        num_workers: Number of worker processes
        image_size: Target image size
        augmentation_level: 'light', 'medium', 'heavy'
        use_weighted_sampling: Whether to use weighted sampling for class balance
        cache_images: Whether to cache images in memory
        validate_data: Whether to validate data integrity
        include_negative_samples: Whether to include images without annotations as negative samples
        handle_missing_annotations: How to handle missing annotations ('warn', 'skip', 'silent')
        augmentation_multiplier: Number of additional augmented versions per image (0=no extra, 3=1:3 ratio)
        deterministic_augmentation: Whether to use deterministic augmentation seeds
    """

    # Define transforms
    train_transform = get_enhanced_transforms(
        image_size=image_size,
        is_training=True,
        augmentation_level=augmentation_level
    )
    val_transform = get_enhanced_transforms(
        image_size=image_size,
        is_training=False
    )

    # Create datasets
    logger.info("Creating training dataset...")
    train_dataset = EnhancedSegmentationDataset(
        root_dir=root_dir,
        split='train',
        transform=train_transform,
        image_size=image_size,
        cache_images=cache_images,
        validate_data=validate_data,
        include_negative_samples=include_negative_samples,
        handle_missing_annotations=handle_missing_annotations,
        augmentation_multiplier=augmentation_multiplier,
        deterministic_augmentation=deterministic_augmentation
    )

    logger.info("Creating validation dataset...")
    valid_dataset = EnhancedSegmentationDataset(
        root_dir=root_dir,
        split='valid',
        transform=val_transform,
        image_size=image_size,
        cache_images=cache_images,
        validate_data=validate_data,
        include_negative_samples=include_negative_samples,
        handle_missing_annotations=handle_missing_annotations,
        augmentation_multiplier=augmentation_multiplier,
        deterministic_augmentation=deterministic_augmentation
    )

    logger.info("Creating test dataset...")
    test_dataset = EnhancedSegmentationDataset(
        root_dir=root_dir,
        split='test',
        transform=val_transform,
        image_size=image_size,
        cache_images=cache_images,
        validate_data=validate_data,
        include_negative_samples=include_negative_samples,
        handle_missing_annotations=handle_missing_annotations,
        augmentation_multiplier=augmentation_multiplier,
        deterministic_augmentation=deterministic_augmentation
    )

    # Create samplers
    train_sampler = None
    if use_weighted_sampling:
        logger.info("Creating weighted sampler for balanced training...")
        train_sampler = create_weighted_sampler(train_dataset)

    # Create data loaders
    loaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(train_sampler is None),  # Don't shuffle if using sampler
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=num_workers > 0
        ),
        'valid': DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0
        ),
        'test': DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0
        )
    }

    # Print dataset information
    logger.info(f"Dataset loaded successfully!")
    logger.info(f"Train: {len(train_dataset)} samples")
    logger.info(f"Valid: {len(valid_dataset)} samples")
    logger.info(f"Test: {len(test_dataset)} samples")
    logger.info(f"Image size: {image_size}x{image_size}")
    logger.info(f"Augmentation level: {augmentation_level}")
    logger.info(f"Weighted sampling: {use_weighted_sampling}")

    return loaders


def get_aggressive_augmentation_transforms(image_size: int = 640, is_training: bool = True) -> A.Compose:
    """
    Get very aggressive augmentation transforms for 1:3 augmentation ratio
    Designed specifically for medical imaging with proper mask alignment
    """

    if not is_training:
        # Validation/test transforms - only normalization
        return A.Compose([
            A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ])

    # Very aggressive training transforms for maximum data diversity
    return A.Compose([
        A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),

        # Geometric transformations (high probability for diversity)
        A.HorizontalFlip(p=0.7),
        A.VerticalFlip(p=0.4),
        A.RandomRotate90(p=0.6),
        A.Transpose(p=0.3),

        # Advanced geometric transformations
        A.ShiftScaleRotate(
            shift_limit=0.2, scale_limit=0.2, rotate_limit=30,
            border_mode=cv2.BORDER_REFLECT_101, p=0.8
        ),
        A.ElasticTransform(
            alpha=1, sigma=50, alpha_affine=50,
            border_mode=cv2.BORDER_REFLECT_101, p=0.4
        ),
        A.GridDistortion(
            num_steps=5, distort_limit=0.15,
            border_mode=cv2.BORDER_REFLECT_101, p=0.3
        ),
        A.OpticalDistortion(
            distort_limit=0.15, shift_limit=0.15,
            border_mode=cv2.BORDER_REFLECT_101, p=0.3
        ),

        # Photometric augmentations (aggressive but medical-safe)
        A.RandomBrightnessContrast(
            brightness_limit=0.4, contrast_limit=0.4, p=0.8
        ),
        A.OneOf([
            A.HueSaturationValue(
                hue_shift_limit=20, sat_shift_limit=40, val_shift_limit=20, p=1.0
            ),
            A.CLAHE(clip_limit=4.0, p=1.0),
            A.RandomGamma(gamma_limit=(60, 140), p=1.0),
            A.ChannelShuffle(p=1.0),
        ], p=0.6),

        # Noise and blur (moderate to preserve medical details)
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 100.0), p=1.0),
            A.GaussianBlur(blur_limit=7, p=1.0),
            A.MotionBlur(blur_limit=7, p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.5),

        # Cutout and occlusion (careful with medical images)
        A.CoarseDropout(
            max_holes=8, max_height=32, max_width=32,
            min_holes=2, min_height=8, min_width=8, p=0.3
        ),

        # Advanced augmentations for more diversity
        A.RandomResizedCrop(
            size=(image_size, image_size),
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
            p=0.3
        ),
        A.PadIfNeeded(
            min_height=image_size, min_width=image_size,
            border_mode=cv2.BORDER_REFLECT_101, p=0.2
        ),

        # Final normalization
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])


def get_1_to_3_augmentation_loaders(
    root_dir: str,
    batch_size: int = 4,
    num_workers: int = 4,
    image_size: int = 640,
    use_weighted_sampling: bool = True,
    cache_images: bool = False,
    validate_data: bool = True,
    include_negative_samples: bool = True,
    handle_missing_annotations: str = 'warn',
    deterministic_augmentation: bool = True
) -> Dict[str, DataLoader]:
    """
    Create enhanced data loaders with 1:3 augmentation ratio
    Each original image gets 3 additional augmented versions (4x total data)

    Args:
        root_dir: Path to dataset directory
        batch_size: Batch size for training
        num_workers: Number of worker processes
        image_size: Target image size
        use_weighted_sampling: Whether to use weighted sampling for class balance
        cache_images: Whether to cache images in memory
        validate_data: Whether to validate data integrity
        include_negative_samples: Whether to include images without annotations
        handle_missing_annotations: How to handle missing annotations
        deterministic_augmentation: Whether to use deterministic augmentation seeds
    """

    logger.info("🚀 Creating 1:3 Augmentation DataLoaders")
    logger.info("=" * 50)

    # Define transforms with aggressive augmentation for training
    train_transform = get_aggressive_augmentation_transforms(
        image_size=image_size,
        is_training=True
    )
    val_transform = get_aggressive_augmentation_transforms(
        image_size=image_size,
        is_training=False
    )

    # Create datasets with 3x augmentation multiplier
    logger.info("Creating training dataset with 1:3 augmentation...")
    train_dataset = EnhancedSegmentationDataset(
        root_dir=root_dir,
        split='train',
        transform=train_transform,
        image_size=image_size,
        cache_images=cache_images,
        validate_data=validate_data,
        include_negative_samples=include_negative_samples,
        handle_missing_annotations=handle_missing_annotations,
        augmentation_multiplier=3,  # 1:3 ratio
        deterministic_augmentation=deterministic_augmentation
    )

    logger.info("Creating validation dataset with 1:3 augmentation...")
    valid_dataset = EnhancedSegmentationDataset(
        root_dir=root_dir,
        split='valid',
        transform=val_transform,
        image_size=image_size,
        cache_images=cache_images,
        validate_data=validate_data,
        include_negative_samples=include_negative_samples,
        handle_missing_annotations=handle_missing_annotations,
        augmentation_multiplier=3,  # 1:3 ratio
        deterministic_augmentation=deterministic_augmentation
    )

    logger.info("Creating test dataset with 1:3 augmentation...")
    test_dataset = EnhancedSegmentationDataset(
        root_dir=root_dir,
        split='test',
        transform=val_transform,
        image_size=image_size,
        cache_images=cache_images,
        validate_data=validate_data,
        include_negative_samples=include_negative_samples,
        handle_missing_annotations=handle_missing_annotations,
        augmentation_multiplier=3,  # 1:3 ratio
        deterministic_augmentation=deterministic_augmentation
    )

    # Create samplers
    train_sampler = None
    if use_weighted_sampling:
        logger.info("Creating weighted sampler for balanced training...")
        train_sampler = create_weighted_sampler(train_dataset)

    # Create data loaders
    loaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=num_workers > 0
        ),
        'valid': DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0
        ),
        'test': DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0
        )
    }

    # Print dataset information
    logger.info("🎉 1:3 Augmentation DataLoaders created successfully!")
    logger.info("=" * 50)
    logger.info(f"📊 Dataset Statistics:")
    logger.info(f"  Original Train: {len(train_dataset.image_ids)} images")
    logger.info(f"  Augmented Train: {len(train_dataset)} samples (4x)")
    logger.info(f"  Original Valid: {len(valid_dataset.image_ids)} images")
    logger.info(f"  Augmented Valid: {len(valid_dataset)} samples (4x)")
    logger.info(f"  Original Test: {len(test_dataset.image_ids)} images")
    logger.info(f"  Augmented Test: {len(test_dataset)} samples (4x)")
    logger.info(f"🖼️  Image size: {image_size}x{image_size}")
    logger.info(f"🔄 Augmentation: 1:3 ratio (aggressive)")
    logger.info(f"⚖️  Weighted sampling: {use_weighted_sampling}")
    logger.info(f"🎯 Deterministic augmentation: {deterministic_augmentation}")

    return loaders


def visualize_dataset_samples(
    dataset: EnhancedSegmentationDataset,
    num_samples: int = 8,
    save_path: Optional[str] = None
) -> None:
    """Visualize random samples from the dataset"""

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 3 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    # Get random samples
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    for i, idx in enumerate(indices):
        sample = dataset[idx]

        # Get image and mask
        image = sample['image']
        mask = sample['mask']
        filename = sample['filename']

        # Convert tensors to numpy for visualization
        if torch.is_tensor(image):
            if image.shape[0] == 3:  # CHW format
                image = image.permute(1, 2, 0).numpy()
            # Denormalize if normalized
            if image.max() <= 1.0:
                image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                image = np.clip(image, 0, 1)

        if torch.is_tensor(mask):
            mask = mask.squeeze().numpy()

        # Plot original image
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f'Image: {filename[:20]}...')
        axes[i, 0].axis('off')

        # Plot mask
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title('Ground Truth Mask')
        axes[i, 1].axis('off')

        # Plot overlay
        overlay = image.copy()
        if len(overlay.shape) == 3:
            mask_colored = np.zeros_like(overlay)
            mask_colored[:, :, 0] = mask  # Red channel for mask
            overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)

        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title('Overlay')
        axes[i, 2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Dataset visualization saved to {save_path}")

    plt.show()


def analyze_dataset_statistics(loaders: Dict[str, DataLoader]) -> Dict:
    """Analyze and print comprehensive dataset statistics"""

    stats = {}

    for split_name, loader in loaders.items():
        logger.info(f"\nAnalyzing {split_name} split...")

        dataset = loader.dataset
        split_stats = {
            'num_samples': len(dataset),
            'positive_ratios': [],
            'mask_areas': [],
            'image_sizes': []
        }

        # Sample a subset for analysis
        sample_size = min(50, len(dataset))
        indices = np.random.choice(len(dataset), sample_size, replace=False)

        for idx in indices:
            try:
                sample = dataset[idx]
                mask = sample['mask']

                if torch.is_tensor(mask):
                    mask = mask.numpy()

                # Calculate statistics
                total_pixels = mask.size
                positive_pixels = np.sum(mask > 0)
                pos_ratio = positive_pixels / total_pixels

                split_stats['positive_ratios'].append(pos_ratio)
                split_stats['mask_areas'].append(positive_pixels)
                split_stats['image_sizes'].append(sample['original_size'])

            except Exception as e:
                logger.warning(f"Error analyzing sample {idx}: {str(e)}")
                continue

        # Calculate summary statistics
        if split_stats['positive_ratios']:
            split_stats['mean_positive_ratio'] = np.mean(split_stats['positive_ratios'])
            split_stats['std_positive_ratio'] = np.std(split_stats['positive_ratios'])
            split_stats['mean_mask_area'] = np.mean(split_stats['mask_areas'])
            split_stats['std_mask_area'] = np.std(split_stats['mask_areas'])
        else:
            split_stats['mean_positive_ratio'] = 0
            split_stats['std_positive_ratio'] = 0
            split_stats['mean_mask_area'] = 0
            split_stats['std_mask_area'] = 0

        stats[split_name] = split_stats

        # Print statistics
        logger.info(f"{split_name.capitalize()} Statistics:")
        logger.info(f"  Samples: {split_stats['num_samples']}")
        logger.info(f"  Mean positive ratio: {split_stats['mean_positive_ratio']:.4f} ± {split_stats['std_positive_ratio']:.4f}")
        logger.info(f"  Mean mask area: {split_stats['mean_mask_area']:.1f} ± {split_stats['std_mask_area']:.1f} pixels")

    return stats


def test_dataloader(root_dir: str, image_size: int = 640) -> None:
    """Test the dataloader with a small batch"""

    logger.info("Testing enhanced dataloader...")

    try:
        # Create loaders with validation
        loaders = get_enhanced_loaders(
            root_dir=root_dir,
            batch_size=2,
            num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
            image_size=image_size,
            augmentation_level='medium',
            use_weighted_sampling=False,
            cache_images=False,
            validate_data=True
        )

        # Test each split
        for split_name, loader in loaders.items():
            logger.info(f"\nTesting {split_name} loader...")

            # Get one batch
            batch = next(iter(loader))

            logger.info(f"  Batch size: {batch['image'].shape[0]}")
            logger.info(f"  Image shape: {batch['image'].shape}")
            logger.info(f"  Mask shape: {batch['mask'].shape}")
            logger.info(f"  Image range: [{batch['image'].min():.3f}, {batch['image'].max():.3f}]")
            logger.info(f"  Mask range: [{batch['mask'].min():.3f}, {batch['mask'].max():.3f}]")
            logger.info(f"  Sample filenames: {batch['filename']}")

        # Analyze dataset statistics
        stats = analyze_dataset_statistics(loaders)

        # Visualize samples
        logger.info("\nVisualizing dataset samples...")
        visualize_dataset_samples(loaders['train'].dataset, num_samples=4)

        logger.info("✅ Dataloader test completed successfully!")

        return loaders

    except Exception as e:
        logger.error(f"❌ Dataloader test failed: {str(e)}")
        raise


if __name__ == "__main__":
    # Test the dataloader
    dataset_path = "Dataset_no_preprocessing"

    if os.path.exists(dataset_path):
        test_dataloader(dataset_path, image_size=640)
    else:
        logger.error(f"Dataset path not found: {dataset_path}")
        logger.info("Please update the dataset_path variable to point to your dataset directory")
