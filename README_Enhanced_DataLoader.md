# Enhanced DataLoader for U-Net Segmentation

This enhanced dataloader is specifically designed for your `Dataset_no_preprocessing` with COCO format annotations. It provides comprehensive preprocessing, advanced augmentation, and robust error handling for U-Net training.

## 🚀 Features

### ✨ Advanced Preprocessing
- **Comprehensive normalization** with ImageNet statistics
- **Intelligent resizing** with aspect ratio preservation
- **Data validation** to catch corrupted samples
- **Memory-efficient loading** with optional caching

### 🔄 Enhanced Augmentation
- **Three intensity levels**: Light, Medium, Heavy
- **Geometric transformations**: Rotation, scaling, flipping, elastic deformation
- **Photometric augmentations**: Brightness, contrast, hue, saturation
- **Advanced techniques**: Cutout, GridDistortion, OpticalDistortion
- **Medical imaging optimized**: Preserves important features

### ⚖️ Class Balancing
- **Weighted sampling** for imbalanced datasets
- **Automatic class weight calculation**
- **Smart sample weighting** based on mask content

### 🛡️ Robust Error Handling
- **Data validation** during loading
- **Graceful error recovery** for corrupted samples
- **Comprehensive logging** for debugging
- **Memory usage optimization**

## 📁 File Structure

```
Mini_Project/u-net/
├── enhanced_dataloader.py          # Main enhanced dataloader
├── test_enhanced_dataloader.py     # Test script
├── integration_guide.py            # Integration instructions
├── README_Enhanced_DataLoader.md   # This file
└── Dataset_no_preprocessing/       # Your dataset
    ├── train/
    │   ├── _annotations.coco.json
    │   └── *.jpg
    ├── valid/
    │   ├── _annotations.coco.json
    │   └── *.jpg
    └── test/
        ├── _annotations.coco.json
        └── *.jpg
```

## 🔧 Installation

1. **Install dependencies**:
```bash
pip install torch torchvision albumentations opencv-python pycocotools matplotlib seaborn pandas
```

2. **Copy the enhanced dataloader**:
```bash
# The files are already in your Mini_Project/u-net/ directory
```

## 🚀 Quick Start

### 1. Test the DataLoader
```python
python test_enhanced_dataloader.py
```

### 2. Basic Usage
```python
from enhanced_dataloader import get_enhanced_loaders

# Create dataloaders
loaders = get_enhanced_loaders(
    root_dir="Dataset_no_preprocessing",
    batch_size=4,
    num_workers=4,
    image_size=640,
    augmentation_level='medium',
    use_weighted_sampling=True,
    cache_images=False,
    validate_data=True
)

# Use in training
for batch in loaders['train']:
    images = batch['image']  # Shape: [B, 3, H, W]
    masks = batch['mask']    # Shape: [B, 1, H, W]
    # ... training code
```

### 3. Integration with Existing Training
```python
# Replace this line in your train_unetpp.py:
# from utils.dataset import get_loaders

# With this:
from enhanced_dataloader import get_enhanced_loaders

# Replace the get_loaders call with:
loaders = get_enhanced_loaders(
    root_dir=args.data_dir,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    image_size=args.image_size,
    augmentation_level='medium',
    use_weighted_sampling=True
)
```

## ⚙️ Configuration Options

### Augmentation Levels

| Level | Description | Use Case |
|-------|-------------|----------|
| `light` | Basic augmentations | Large datasets, stable training |
| `medium` | Balanced augmentations | Most use cases, good performance |
| `heavy` | Intensive augmentations | Small datasets, need more variation |

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `root_dir` | str | - | Path to dataset directory |
| `batch_size` | int | 4 | Batch size for training |
| `num_workers` | int | 4 | Number of worker processes |
| `image_size` | int | 640 | Target image size (square) |
| `augmentation_level` | str | 'medium' | Augmentation intensity |
| `use_weighted_sampling` | bool | False | Enable weighted sampling |
| `cache_images` | bool | False | Cache images in memory |
| `validate_data` | bool | True | Validate data integrity |

## 📊 Recommended Configurations

### Small Dataset (< 1000 images)
```python
loaders = get_enhanced_loaders(
    root_dir="Dataset_no_preprocessing",
    batch_size=8,
    image_size=512,
    augmentation_level='heavy',
    use_weighted_sampling=True,
    cache_images=True,
    validate_data=True
)
```

### Medium Dataset (1000-5000 images)
```python
loaders = get_enhanced_loaders(
    root_dir="Dataset_no_preprocessing",
    batch_size=4,
    image_size=640,
    augmentation_level='medium',
    use_weighted_sampling=True,
    cache_images=False,
    validate_data=True
)
```

### Large Dataset (> 5000 images)
```python
loaders = get_enhanced_loaders(
    root_dir="Dataset_no_preprocessing",
    batch_size=2,
    image_size=768,
    augmentation_level='light',
    use_weighted_sampling=False,
    cache_images=False,
    validate_data=False
)
```

## 🔍 Dataset Analysis

The dataloader provides comprehensive dataset analysis:

```python
from enhanced_dataloader import analyze_dataset_statistics, visualize_dataset_samples

# Analyze statistics
stats = analyze_dataset_statistics(loaders)

# Visualize samples
visualize_dataset_samples(
    loaders['train'].dataset, 
    num_samples=8,
    save_path="dataset_samples.png"
)
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Error**: Make sure all dependencies are installed
2. **COCO Error**: Verify annotation files are valid JSON
3. **Memory Error**: Reduce batch_size or disable cache_images
4. **Slow Loading**: Reduce num_workers or disable data validation

### Debug Mode
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Test with minimal configuration
loaders = get_enhanced_loaders(
    root_dir="Dataset_no_preprocessing",
    batch_size=1,
    num_workers=0,
    validate_data=True
)
```

## 📈 Performance Tips

1. **Batch Size**: Start with 4, adjust based on GPU memory
2. **Workers**: Use 2-4 workers for most systems
3. **Image Size**: 640 is a good balance, use 512 for faster training
4. **Caching**: Only enable for datasets < 1GB
5. **Validation**: Disable for large datasets to speed up loading

## 🔄 Comparison with Original

| Feature | Original | Enhanced |
|---------|----------|----------|
| Augmentations | Basic (flip only) | Comprehensive (15+ types) |
| Error Handling | Minimal | Robust with recovery |
| Class Balance | None | Weighted sampling |
| Data Validation | None | Comprehensive checks |
| Memory Efficiency | Basic | Optimized with caching |
| Preprocessing | Basic resize | Advanced with normalization |
| Logging | None | Detailed logging |

## 🎯 Next Steps

1. **Test the dataloader**: Run `python test_enhanced_dataloader.py`
2. **Integrate with training**: Follow the integration guide
3. **Experiment with configurations**: Try different augmentation levels
4. **Monitor training**: Use the enhanced metrics and visualizations
5. **Optimize performance**: Adjust parameters based on your hardware

## 📞 Support

If you encounter any issues:
1. Check the troubleshooting section
2. Run the test script for debugging
3. Enable detailed logging
4. Verify your dataset structure matches the expected format

The enhanced dataloader is designed to be a drop-in replacement for the original dataloader while providing significantly more features and robustness.
