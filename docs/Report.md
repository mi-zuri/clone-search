# Face Search & Inpainting - Project Report

## Project Overview

Multi-task deep learning system for face similarity search and semantic inpainting.

**Grade Target:** 5 (13 points base, 15 with Grad-CAM)

## Point Breakdown

| Category | Item | Points |
|----------|------|--------|
| Problem | Inpainting | 3 |
| Problem | Search Engine | 2 |
| Model | Own architecture (>50% custom) | 2 |
| Model | Non-trivial (multi-task learning) | +1 |
| Dataset | Eval >10k images (70k FFHQ) | +1 |
| Training | Data augmentation | +1 |
| Training | Hyperparameter estimation | +1 |
| Tools | TensorBoard | +1 |
| Tools | Streamlit GUI | +1 |
| **Total (base)** | | **13** |
| Tools | Grad-CAM explainability | +2 |
| **Total (max)** | | **15** |

---

## Datasets

### CelebAMask-HQ (Training)
- **Size:** 30,000 images
- **Resolution:** 256×256 (resized from 512×512)
- **Labels:** 40 binary attributes
- **Masks:** Semantic segmentation (skin, eyes, nose, mouth, etc.)
- **Split:** 24k train / 3k val / 3k test

### FFHQ (Search Gallery)
- **Size:** 70,000 images
- **Resolution:** 256×256
- **Purpose:** Search gallery for face retrieval

---

## Model Architectures

### Face Encoder

Custom 6-layer CNN with dual heads:

```
Input: (B, 3, 256, 256)
├── Conv2d(3, 64, 7, stride=2) + BN + ReLU + MaxPool  → (B, 64, 64, 64)
├── Conv2d(64, 128, 3) + BN + ReLU                    → (B, 128, 64, 64)
├── Conv2d(128, 128, 3, stride=2) + BN + ReLU         → (B, 128, 32, 32)
├── Conv2d(128, 256, 3) + BN + ReLU                   → (B, 256, 32, 32)
├── Conv2d(256, 256, 3, stride=2) + BN + ReLU         → (B, 256, 16, 16)
├── Conv2d(256, 512, 3) + BN + ReLU                   → (B, 512, 16, 16)
├── GlobalAvgPool                                      → (B, 512)
├── Head A: FC(512, 128) → embedding (L2 normalized)
└── Head B: FC(512, 40) → attribute logits
```

**Parameters:** ~5.2M

### U-Net Inpainter

Standard U-Net with 4-channel input (RGB + mask):

```
Encoder:
├── DoubleConv(4, 64)   → skip1
├── Down(64, 128)       → skip2
├── Down(128, 256)      → skip3
├── Down(256, 512)      → skip4
└── Down(512, 512)      → bottleneck

Decoder:
├── Up(1024, 256) + skip4
├── Up(512, 128) + skip3
├── Up(256, 64) + skip2
├── Up(128, 64) + skip1
└── Conv(64, 3) + Sigmoid
```

**Parameters:** ~7.8M

---

## Training Details

### Encoder Training

| Hyperparameter | Value |
|---------------|-------|
| Batch size | 32 |
| Epochs | 10 |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Optimizer | Adam |
| Scheduler | CosineAnnealingLR |
| Loss | BCEWithLogitsLoss |

### U-Net Training

| Hyperparameter | Value |
|---------------|-------|
| Batch size | 8 |
| Epochs | 15 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Scheduler | CosineAnnealingLR |
| Loss | L1 with mask weighting (6×) |

### Data Augmentation

- HorizontalFlip (p=0.5)
- ShiftScaleRotate (shift=0.05, scale=0.05, rotate=10°)
- ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
- GaussNoise (var=10-50)
- ImageNet normalization

---

## Results

### Encoder

| Metric | Train | Val | Test |
|--------|-------|-----|------|
| BCE Loss | - | - | - |
| Mean Accuracy | - | - | - |

*Fill in after training*

### U-Net Inpainter

| Metric | Train | Val | Test |
|--------|-------|-----|------|
| L1 Loss | - | - | - |
| PSNR (dB) | - | - | - |
| SSIM | - | - | - |

*Fill in after training*

---

## Architecture Diagrams

### System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Streamlit GUI                            │
├─────────────────────────────┬───────────────────────────────────┤
│       Face Search Tab       │         Inpainting Tab            │
└─────────────────────────────┴───────────────────────────────────┘
              │                            │
              ▼                            ▼
┌─────────────────────────┐    ┌─────────────────────────┐
│      Face Encoder       │    │        U-Net            │
│  ┌───────────────────┐  │    │                         │
│  │ Conv Backbone     │  │    │  ┌──────────────────┐   │
│  │ (6 layers)        │  │    │  │ Encoder (4 down) │   │
│  └────────┬──────────┘  │    │  └────────┬─────────┘   │
│           │             │    │           │             │
│  ┌────────┴──────────┐  │    │  ┌────────┴─────────┐   │
│  │ Global Avg Pool   │  │    │  │ Bottleneck       │   │
│  └────────┬──────────┘  │    │  └────────┬─────────┘   │
│           │             │    │           │             │
│     ┌─────┴─────┐       │    │  ┌────────┴─────────┐   │
│     ▼           ▼       │    │  │ Decoder (4 up)   │   │
│ [Embed]    [Attrs]      │    │  └────────┬─────────┘   │
│  128-d      40-d        │    │           │             │
└─────────────────────────┘    │     RGB Output         │
              │                └─────────────────────────┘
              ▼
┌─────────────────────────┐
│    FAISS Search Index   │
│    (70k FFHQ images)    │
└─────────────────────────┘
```

### Encoder Architecture

```
Input Image (256×256×3)
         │
         ▼
   ┌───────────┐
   │  Conv 7×7 │ stride=2, out=64
   │    + BN   │
   │   + ReLU  │
   │ + MaxPool │
   └─────┬─────┘
         │ 64×64×64
         ▼
   ┌───────────┐
   │  Conv 3×3 │ out=128
   │    + BN   │
   │   + ReLU  │
   └─────┬─────┘
         │ 64×64×128
         ▼
   ┌───────────┐
   │  Conv 3×3 │ stride=2, out=128
   │    + BN   │
   │   + ReLU  │
   └─────┬─────┘
         │ 32×32×128
         ▼
   ┌───────────┐
   │  Conv 3×3 │ out=256
   │    + BN   │
   │   + ReLU  │
   └─────┬─────┘
         │ 32×32×256
         ▼
   ┌───────────┐
   │  Conv 3×3 │ stride=2, out=256
   │    + BN   │
   │   + ReLU  │
   └─────┬─────┘
         │ 16×16×256
         ▼
   ┌───────────┐
   │  Conv 3×3 │ out=512
   │    + BN   │
   │   + ReLU  │
   └─────┬─────┘
         │ 16×16×512
         ▼
   ┌───────────┐
   │ GlobalAvg │
   │   Pool    │
   └─────┬─────┘
         │ 512
         ▼
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌───────┐
│FC 128 │ │FC 40  │
│L2 Norm│ │Sigmoid│
└───────┘ └───────┘
Embedding  Attributes
```

---

## Usage

### Training

```bash
# Train encoder
python -m src.training.train_encoder --data-dir data/CelebAMask-HQ

# Train U-Net
python -m src.training.train_unet --data-dir data/CelebAMask-HQ

# Build search index
python -m src.search.engine --model checkpoints/encoder_best.pth --gallery data/FFHQ
```

### Streamlit App

```bash
streamlit run src/app.py
```

### TensorBoard

```bash
tensorboard --logdir runs
```

---

## References

1. CelebAMask-HQ Dataset: Lee et al., "MaskGAN: Towards Diverse and Interactive Facial Image Manipulation"
2. FFHQ Dataset: Karras et al., "A Style-Based Generator Architecture for Generative Adversarial Networks"
3. U-Net: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation"
