# RainPredRNN

A PyTorch implementation of spatiotemporal recurrent neural networks for radar-based rainfall prediction. This project implements multiple variants of PredRNN models that learn to predict future rainfall radar images from historical sequences using dual-memory LSTM architectures.

## 🌟 Features

- **Multiple Model Architectures**: Implementation of 5 different spatiotemporal LSTM variants
  - PredRNN: Baseline with SpatioTemporalLSTMCell
  - PredRNN-v2: Enhanced with zigzag memory flow
  - Attention-PredRNN: Attention-based encoder-decoder
  - RainPredRNN: UNet encoder + LSTM + UNet decoder
  - RainPredRNN-v2: UNet + enhanced LSTM with decoupling loss

- **Dual Memory Mechanism**: Captures both temporal dynamics (cell state `c_t`) and spatial patterns (spatial memory `m_t`)

- **Comprehensive Evaluation**: MSE, SSIM, and CSI (Critical Success Index) metrics

- **Visualization Tools**: Generate comparison plots across different models

- **Model Analysis**: Compute MACs and parameter counts using `thop`

## 📋 Requirements

```bash
torch==2.6.0
tqdm==4.67.1
Pillow==11.1.0
numpy==2.2.3
tensorboardX==2.2
matplotlib==3.10.1
thop==0.1.1.post2209072238
scikit-image==0.25.2
scikit-learn==1.6.1
```

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rainpredrnn.git
cd rainpredrnn
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Dataset Preparation

### Dataset Format
- **Type**: Sequential radar rainfall images
- **Format**: Grayscale images (PNG/JPEG)
- **Naming**: Files must be named to sort chronologically (e.g., `PHA200623065004.RAWV7AW.png`)
- **Organization**: Place all images in a single directory

### Dataset Structure
```
dataset/
└── timeseries/
    └── dataset/
        ├── 20200623-20200915/  # Training data
        │   ├── PHA200623065004.RAWV7AW.png
        │   ├── PHA200623065504.RAWV7AW.png
        │   └── ...
        └── 20200603/           # Test data
            ├── PHA200603065004.RAWV7AW.png
            └── ...
```

The dataset loader will:
- Convert images to grayscale
- Resize to `img_width x img_width` (default: 128x128)
- Normalize to float32 [0, 1]
- Create sliding windows of length `total_length`

## 🎯 Usage

### Training

Edit the `Config` class in `src/train.py` to set hyperparameters:

```python
class Config():
    model_name = 'predrnn'  # Choose: predrnn, predrnn_v2, attention_predrnn, rainpredrnn, rainpredrnn_v2
    path_root = '../dataset/timeseries/dataset/20200623-20200915'
    
    # Model parameters
    channel = 1
    num_hidden = [64]
    layer_norm = True
    img_width = 128
    filter_size = 3
    
    # Sequence parameters
    total_length = 10
    input_length = 5
    pred_length = 5
    
    # Training parameters
    batch_size = 2
    epochs = 100
    lr = 0.001
    device = 'cuda'
```

Run training:
```bash
python src/train.py
```

**Output:**
- Checkpoints saved to `checkpoints/{model_name}/model-{epoch}.ckpt`
- TensorBoard logs in `logs/`

View training progress:
```bash
tensorboard --logdir=logs
```

### Testing

Edit `src/test.py` to configure the test:

```python
configs = Config()
configs.path_root = '../dataset/timeseries/dataset/20200603'
configs.model_name = 'predrnn'

model = Model(configs)
model = load_model(model, 'checkpoints/predrnn/model-95.ckpt')
```

Run evaluation:
```bash
python src/test.py
```

**Metrics computed:**
- MSE (Mean Squared Error)
- SSIM (Structural Similarity Index)
- CSI (Critical Success Index) - uses threshold 0.12 for rain/no-rain classification

### Prediction & Visualization

Generate visual comparisons:

```bash
python src/predict.py
```

**Output:** Saves comparison plots to `result_images/` showing:
- Input sequence (t-5 to t)
- Ground truth predictions (t+1 to t+5)
- Model predictions for each architecture

### Model Complexity Analysis

Compute MACs and parameters:

```python
# Edit src/MACs.py to select model
configs.model_name = 'rainpredrnn_v2'
```

```bash
python src/MACs.py
```

## 🏗️ Architecture

### Model Factory Pattern

All models are instantiated through `src/models/factory.py`:

```python
from src.models.factory import Model

configs = Config()
model = Model(configs)  # Automatically selects based on configs.model_name
```

### Tensor Format Convention

**Critical:** Models expect specific tensor formats:

```python
# Input format:  (batch, length, height, width, channel)
# Internal:      (batch, length, channel, height, width)  # After permutation
# Output format: (batch, length, height, width, channel)  # Permuted back
```

Models handle permutation internally at boundaries.

### Scheduled Sampling

During training, the model uses scheduled sampling:

```python
if t < configs.input_length:
    net = frames[:, t]  # Use ground truth
else:
    # Blend ground truth and prediction based on mask
    net = mask_true[:, t - configs.input_length] * frames[:, t] + \
          (1 - mask_true[:, t - configs.input_length]) * x_gen
```

- Training: `mask_true` is zeros → uses only predictions
- Testing: Can inject ground truth for analysis

## 📁 Project Structure

```
rainpredrnn/
├── src/
│   ├── models/                    # Model implementations
│   │   ├── factory.py            # Model factory
│   │   ├── predrnn.py            # Baseline PredRNN
│   │   ├── predrnn_v2.py         # PredRNN with zigzag memory
│   │   ├── attention_predrnn.py  # Attention-based variant
│   │   ├── rainpredrnn.py        # UNet + LSTM
│   │   └── rainpredrnn_v2.py     # UNet + enhanced LSTM
│   ├── layers/                    # Network components
│   │   ├── spatio_temporal_lstm_cell.py      # Basic ST-LSTM
│   │   ├── spatio_temporal_lstm_cell_v2.py   # Enhanced ST-LSTM
│   │   ├── unet_cell.py                      # UNet encoder/decoder
│   │   └── attention_encoder_decoder.py      # Attention mechanism
│   ├── dataset/                   # Data loading
│   │   ├── get_data.py           # Image loader
│   │   └── dataset.py            # Sliding window dataset
│   ├── train.py                   # Training script
│   ├── test.py                    # Evaluation script
│   ├── predict.py                 # Visualization script
│   ├── MACs.py                    # Model complexity analysis
│   ├── utils.py                   # Save/load utilities
│   └── viz.py                     # Additional visualization
├── checkpoints/                   # Saved model checkpoints
├── logs/                          # TensorBoard logs
├── result_images/                 # Prediction visualizations
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🔧 Configuration

Key configuration parameters in `Config` class:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model_name` | Model architecture to use | `'predrnn'` |
| `path_root` | Dataset directory path | `'../dataset/...'` |
| `channel` | Number of input channels | `1` |
| `num_hidden` | Hidden layer sizes | `[64]` |
| `layer_norm` | Use layer normalization | `True` |
| `img_width` | Image resize dimension | `128` |
| `filter_size` | Convolution kernel size | `3` |
| `total_length` | Total sequence length | `10` |
| `input_length` | Input frames count | `5` |
| `pred_length` | Prediction frames count | `5` |
| `batch_size` | Training batch size | `2` |
| `epochs` | Training epochs | `100` |
| `lr` | Learning rate | `0.001` |
| `device` | Training device | `'cuda'` |
| `split_data` | Train/test split ratio | `0.8` |
| `eval_interval` | Evaluation frequency | `1` |
| `save_interval` | Checkpoint save frequency | `5` |

**Note:** Always ensure `total_length = input_length + pred_length`

## 🎓 Adding New Models

To add a custom model:

1. Create `src/models/your_model.py`:
```python
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(RNN, self).__init__()
        # Your architecture here
        
    def forward(self, frames, mask_true):
        # frames: (batch, length, height, width, channel)
        # Returns: (batch, length-1, height, width, channel)
        pass
```

2. Register in `src/models/factory.py`:
```python
networks_map = {
    'predrnn': predrnn.RNN,
    'your_model': your_model.RNN,  # Add here
}
```

3. Follow conventions:
   - Permute input from BHWC to BCHW
   - Initialize states as zeros on `configs.device`
   - Permute output back to BHWC

## 📊 Evaluation Metrics

### MSE (Mean Squared Error)
Standard pixel-wise reconstruction error

### SSIM (Structural Similarity Index)
Perceptual similarity metric considering luminance, contrast, and structure

### CSI (Critical Success Index)
Binary classification metric for rain detection:
- Threshold: 0.12 for rain/no-rain classification
- Formula: `hits / (hits + misses + false_alarms)`

## ⚠️ Common Issues

| Issue | Solution |
|-------|----------|
| Dimension mismatch errors | Check BHWC vs BCHW format before Conv2d operations |
| CUDA out of memory | Reduce `batch_size` or `img_width` |
| Checkpoint not found | Verify path and ensure model was saved with `save_interval` |
| Model name not recognized | Check `model_name` matches `networks_map` in `factory.py` |
| Dataset not loading | Ensure images sort chronologically by filename |

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{tuyen2022rainpredrnn,
  title={RainPredRNN: A new approach for precipitation nowcasting with weather radar echo images based on deep learning},
  author={Tuyen, Do Ngoc and Tuan, Tran Manh and Le, Xuan-Hien and Tung, Nguyen Thanh and Chau, Tran Kim and Van Hai, Pham and Gerogiannis, Vassilis C and Son, Le Hoang},
  journal={Axioms},
  volume={11},
  number={3},
  pages={107},
  year={2022},
  publisher={MDPI}
}
```
---

**Note**: This implementation is designed for radar-based rainfall prediction but can be adapted for other spatiotemporal prediction tasks by modifying the dataset loader and configuration parameters.
