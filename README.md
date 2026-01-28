# Face Search & Inpainting

Computer Vision Project - Multi-task learning for face similarity search and semantic inpainting.

## Features

- **Face Search**: Find similar faces using learned embeddings + FAISS
- **Face Inpainting**: Remove and reconstruct facial features (eyes, nose, mouth)
- **Attribute Prediction**: 40 CelebA binary attributes
- **Grad-CAM**: Visual explanation of encoder attention

## Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Data

### Download Datasets

1. **CelebAMask-HQ** (Training) - ~3GB:
```bash
# Using Kaggle CLI
kaggle datasets download -d badasstechie/celebahq-resized-256x256
unzip celebahq-resized-256x256.zip -d data/CelebAMask-HQ
```

2. **FFHQ** (Search Gallery) - ~20GB:
```bash
kaggle datasets download -d arnaud58/flickrfaceshq-dataset-ffhq
unzip flickrfaceshq-dataset-ffhq.zip -d data/FFHQ
```

## Training

### 1. Train Encoder
```bash
python -m src.training.train_encoder --data-dir data/CelebAMask-HQ
```

### 2. Train U-Net Inpainter
```bash
python -m src.training.train_unet --data-dir data/CelebAMask-HQ
```

### 3. Build Search Index
```bash
python -m src.search.engine \
    --model checkpoints/encoder_best.pth \
    --gallery data/FFHQ \
    --output checkpoints/gallery_index.npy
```

## Run Application

```bash
streamlit run src/app.py
```

## Monitor Training

```bash
tensorboard --logdir runs
```

## Project Structure

```
CV_clone_search/
├── src/
│   ├── data/
│   │   ├── dataset.py        # CelebA & FFHQ loaders
│   │   └── augmentations.py  # albumentations pipeline
│   ├── models/
│   │   ├── encoder.py        # Face encoder (embedding + attributes)
│   │   └── unet.py           # U-Net inpainter
│   ├── training/
│   │   ├── train_encoder.py  # Encoder training script
│   │   └── train_unet.py     # U-Net training script
│   ├── search/
│   │   └── engine.py         # FAISS search engine
│   └── app.py                # Streamlit GUI
├── configs/
│   └── config.yaml           # Hyperparameters
├── data/                     # Datasets (gitignore)
├── checkpoints/              # Model weights
├── runs/                     # TensorBoard logs
└── docs/
    └── Report.md             # Project report
```

## Configuration

Edit `configs/config.yaml`:

```yaml
encoder:
  batch_size: 32
  epochs: 10
  lr: 0.001

unet:
  batch_size: 8
  epochs: 15
  lr: 0.001

data:
  image_size: 256
```

## Requirements

- Python 3.11+
- PyTorch 2.0+
- CUDA (recommended) or MPS (Apple Silicon)
