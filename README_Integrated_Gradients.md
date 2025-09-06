# Integrated Gradients for U-Net++ EfficientNet-B5 Segmentation Model

This implementation provides **Integrated Gradients** explainability analysis for your trained U-Net++ segmentation model with EfficientNet-B5 encoder. Integrated Gradients helps you understand which pixels the model considers most important when making segmentation predictions.

## 🔬 What is Integrated Gradients?

Integrated Gradients is a method for attributing the prediction of a deep network to its input features. It satisfies two fundamental axioms:

- **Sensitivity**: If two inputs differ in one feature but have different predictions, then the differing feature should have a non-zero attribution
- **Implementation Invariance**: The attributions are always identical for two functionally equivalent networks

For segmentation models, this helps us understand which pixels the model considers most important for making predictions.

## 📁 Files Overview

- `integrated_gradients.py` - Core implementation of Integrated Gradients for segmentation models
- `run_integrated_gradients.py` - Simple script to run analysis on your best model
- `integrated_gradients_interactive.ipynb` - Interactive Jupyter notebook for exploration
- `README_Integrated_Gradients.md` - This documentation file

## 🚀 Quick Start

### Option 1: Simple Script (Recommended)

Run the pre-configured script for your best model:

```bash
python run_integrated_gradients.py
```

This will:
1. Load your best model (`models_20250609_105424/best_model.pth`)
2. Ask you to choose between single image or batch analysis
3. Generate comprehensive visualizations and statistics
4. Save results to `integrated_gradients_results/`

### Option 2: Command Line Interface

For more control, use the main script:

```bash
# Single image analysis
python integrated_gradients.py \
    --model_path models_20250609_105424/best_model.pth \
    --mode single \
    --image_path Dataset_no_preprocessing/test/your_image.jpg \
    --output_dir results \
    --baseline zero \
    --num_steps 50

# Batch analysis
python integrated_gradients.py \
    --model_path models_20250609_105424/best_model.pth \
    --mode batch \
    --data_dir Dataset_no_preprocessing \
    --output_dir results \
    --num_samples 10 \
    --baseline zero \
    --num_steps 50
```

### Option 3: Interactive Jupyter Notebook

For interactive exploration:

```bash
jupyter notebook integrated_gradients_interactive.ipynb
```

## 📊 Understanding the Results

### Visualizations Generated

1. **Main Attribution Visualization**:
   - Original image
   - Attribution heatmap (hot colormap)
   - Model prediction
   - Ground truth (if available)
   - Various overlays

2. **Channel-wise Analysis**:
   - Attribution for each RGB channel
   - Shows which color channels contribute most

3. **Statistics Report**:
   - Attribution statistics (mean, std, max, min)
   - Prediction statistics (area, ratio)
   - Attribution analysis (focus on predicted vs background regions)

### Key Metrics Explained

- **Attribution Mean/Std**: Overall attribution statistics across the image
- **Attribution Ratio**: How much more the model focuses on predicted regions vs background
  - Higher values (>1) indicate good focus on relevant regions
  - Lower values (<1) suggest the model may be looking at irrelevant areas
- **Prediction Ratio**: Proportion of image predicted as positive class

### Baseline Types

- **Zero** (default): Uses all-black image as baseline
- **Random**: Uses random noise as baseline  
- **Blur**: Uses Gaussian-blurred version of input as baseline
- **Mean**: Uses mean pixel values as baseline

Different baselines can reveal different aspects of model behavior.

## 🔧 Configuration Options

### Model Parameters
- `--encoder`: Encoder architecture (default: efficientnet-b5)
- `--image_size`: Input image size (default: 640)

### Integrated Gradients Parameters
- `--baseline`: Baseline type (zero, random, blur, mean)
- `--num_steps`: Number of integration steps (default: 50)
  - More steps = more accurate but slower
  - 50 steps is usually sufficient

### Analysis Parameters
- `--num_samples`: Number of samples for batch analysis (default: 10)
- `--device`: Device to use (auto, cuda, cpu)

## 📈 Interpreting Results

### Good Attribution Patterns
- High attribution in regions where the model predicts positive class
- Low attribution in background regions
- Attribution patterns that align with anatomical features
- High attribution ratio (>2.0)

### Concerning Patterns
- High attribution in background regions
- Low attribution in predicted regions
- Attribution focused on image artifacts or borders
- Low attribution ratio (<1.0)

### Example Interpretation

```
Attribution Statistics:
- Mean: 0.000234 (overall attribution level)
- Std: 0.001456 (attribution variability)
- Attribution Ratio: 3.45 (model focuses 3.45x more on predicted regions)
- Prediction Ratio: 0.12 (12% of image predicted as positive)
```

This shows the model is focusing well on relevant regions (high attribution ratio).

## 🛠️ Advanced Usage

### Custom Analysis

```python
from integrated_gradients import IntegratedGradients, load_model

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model('path/to/model.pth', device, 'efficientnet-b5')

# Initialize IG
ig = IntegratedGradients(model, device)

# Compute attributions
attributions = ig.compute_integrated_gradients(
    input_tensor,
    baseline_type='zero',
    num_steps=50
)
```

### Batch Processing

```python
from integrated_gradients import batch_integrated_gradients_analysis

results = batch_integrated_gradients_analysis(
    model_path='models_20250609_105424/best_model.pth',
    data_dir='Dataset_no_preprocessing',
    output_dir='results',
    num_samples=20,
    baseline_type='blur',
    num_steps=100
)
```

## 🔍 Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `num_steps` or use CPU
2. **Model not found**: Check the model path
3. **No images found**: Verify dataset directory structure
4. **Slow processing**: Reduce `num_steps` or `num_samples`

### Performance Tips

- Use GPU for faster computation
- Start with fewer steps (20-30) for quick exploration
- Use batch analysis for systematic evaluation
- Save results to avoid recomputation

## 📚 References

- Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic attribution for deep networks. ICML.
- [Original Integrated Gradients Paper](https://arxiv.org/abs/1703.01365)

## 🎯 Your Model Performance

Your best model (`models_20250609_105424/best_model.pth`) achieved:
- **Dice Score**: 0.8945
- **IoU Score**: 0.8135
- **Pixel Accuracy**: 0.9972
- **Architecture**: U-Net++ with EfficientNet-B5 encoder

This high-performing model should show meaningful attribution patterns that align with the anatomical structures it's segmenting.
