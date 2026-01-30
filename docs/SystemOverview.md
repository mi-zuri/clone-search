# CV Clone Search - Simplified System Overview

This document provides a complete explanation of the **CV Clone Search** project architecture, components, and execution workflow.

---

## Project Overview

This is a **multi-task face analysis system** that does three things:
1. **Face Search** - Find similar faces using learned embeddings
2. **Face Inpainting** - Reconstruct missing facial regions (eyes, nose, mouth)
3. **Attribute Prediction** - Classify 40 binary facial attributes (e.g., Bald, Smiling, Male)

The project uses **two neural networks** trained sequentially: **FaceEncoder** (MobileNetV2 backbone, 64-dim embeddings) and **LiteUNet** (masked region reconstruction).

**Target Hardware:** M1 MacBook Air (8GB) using MPS backend.

---

## Step-by-Step System Flow

### Step 1: Data Preparation

**Datasets:** CelebAMask-HQ (30k) + FFHQ (~50k) = 80k total images

**Folder Structure:**
```
data/
├── celeba/                              # CelebA-HQ images (30k)
│   ├── 0/ ... 30/                       # 31 subfolders (~1000 images each)
│   │   └── {id}.jpg                     # Naming: 0.jpg, 1.jpg, ..., 29999.jpg
├── celeba_masks/                        # Semantic segmentation masks
│   ├── 0/ ... 372/                      # Mirrors celeba/ structure
│   │   └── {id:05d}_{region}.png        # Naming: 00000_nose.png, 00001_hair.png
├── ffhq/                                # FFHQ images (~50k)
│   ├── 0/ ... 52/                       # 53 subfolders (1000 images each)
│   │   └── {id:05d}.png                 # Naming: 00000.png, 00001.png, ...
└── CelebAMask-HQ-attribute-anno.txt     # 40 binary attributes
```

**Mask Regions (18 types):** `cloth`, `ear_r`, `eye_g`, `hair`, `hat`, `l_brow`, `l_ear`, `l_eye`, `l_lip`, `mouth`, `neck`, `neck_l`, `nose`, `r_brow`, `r_ear`, `r_eye`, `skin`, `u_lip`

**Annotation File Format:**

| File | Line 1 | Line 2 | Lines 3+ |
|------|--------|--------|----------|
| `*-attribute-anno.txt` | `30000` (count) | 40 attribute names (space-sep) | `{filename}  {40 × (-1\|1)}` |

**Example Attribute Row:** `0.jpg  -1 1 1 1 -1 ...` (40 values: -1=absent, 1=present)

**Data Splits:**

| Purpose | Dataset | Split | Count |
|---------|---------|-------|-------|
| Training (supervised: attrs) | CelebA-HQ | train | 15k |
| Validation | CelebA-HQ | val | 5k |
| Testing (attrs, retrieval) | CelebA-HQ | test | 10k |
| Training (SimCLR, no labels needed) | FFHQ | train | 5k (random subset) |
| Gallery + Search Index | CelebA + FFHQ | all | 80k (30k + 50k) |

**Note:** FFHQ is used for **both** training (5k subset for SimCLR) and gallery (full 50k for search). The 5k training subset is randomly sampled; remaining 45k FFHQ images are gallery-only. SimCLR learns from augmentation pairs, so fewer unlabeled samples suffice.

**Sampling:** Balanced sampling to preserve approximate class distribution in reduced training set.

**Batch Composition:** Each training batch contains 75% CelebA + 25% FFHQ samples. SimCLR loss computed on all samples; attribute loss computed on CelebA samples only.

**Data Pipeline (`src/data/dataset.py`):** Raw Images (256×256) → Augmentation (horizontal flip, color jitter) → Normalization → Batching

---

### Step 2: Train the Face Encoder

**Script:** `python -m src.training.train_encoder --config configs/config.yaml`

**Architecture (`src/models/encoder.py`):**
```
Input (256×256 RGB)
    ↓
MobileNetV2 Backbone (pretrained ImageNet)
├── Layers 0-14: Frozen (feature extraction)
└── Layers 15-18: Trainable (task adaptation)
    ↓
Global Average Pooling → 1280-dim vector
    ↓
Split into three heads:
├── Embedding Head: FC(1280→256→64) + L2-normalize → 64-dim (inference)
├── Projection Head: FC(64→128→64) + L2-normalize → 64-dim (SimCLR training only, discarded after)
└── Attribute Head: FC(1280→40) → 40 attribute predictions
```

**Training Details:**
- **Losses:** SimCLR (projection head, self-supervised) + BCEWithLogits with class-weighting (attributes)
- **SimCLR:** No identity labels needed. Positive pairs = two augmentations of the same image; negatives = other images in batch. Uses NT-Xent loss with temperature **τ=0.25**.
- **Total Loss:** `0.5×simclr_loss + 1.0×attribute_loss`
  - Weights calibrated to balance loss magnitudes (~0.5-1.0 each after weighting)
  - Attribute loss computed on CelebA samples only; SimCLR on all samples
- **Optimizer:** AdamW (lr=0.001, weight_decay=0.01) with CosineAnnealingLR + linear warmup (1 epoch, start_lr=1e-5)
- **Gradient Clipping:** max_norm=1.0
- **Epochs:** 5 (with early stopping, patience=2)
- **Validation:** Every epoch
- **Output:** `checkpoints/encoder_best.pth`

**Note:** The projection head (2-layer MLP with expansion) protects the embedding representation from contrastive loss distortion. NT-Xent is applied to the projection output; after training, the projection head is discarded and only the 64-dim embedding head is used for inference.

**M1 MPS Specific:**
- **Batch Size:** 32 (smaller model fits more samples)
- **Gradient Accumulation:** 8 steps (effective batch = 256 for SimCLR)
- **No Mixed Precision:** MPS has limited float16 support; use float32
- **Num Workers:** 2 with spawn method

**TensorBoard Logs:** Learning rate, gradient norm, mean/per-attribute accuracy, embedding L2 norm stats

---

### Step 3: Train the LiteUNet Inpainter

**Script:** `python -m src.training.train_unet --config configs/config.yaml`

**Architecture (`src/models/unet.py`):**
```
Input: 4 channels (3 RGB + 1 mask)
    ↓
Encoder: DoubleConv blocks with MaxPool (32→64→128→256→512)
    ↓
Decoder: Upsample + Skip Connections (512→256→128→64→32)
    ↓
Output Conv + Sigmoid → 3-channel RGB [0,1]
```

**Channel Progression:**

| Stage | Original UNet | LiteUNet | Reduction |
|-------|---------------|----------|-----------|
| Level 1 | 64 | 32 | 2× |
| Level 2 | 128 | 64 | 2× |
| Level 3 | 256 | 128 | 2× |
| Level 4 | 512 | 256 | 2× |
| Bottleneck | 1024 | 512 | 2× |
| **Total Params** | ~31M | ~8M | **4×** |

**Training Details:**
- **Loss:** `L_total = 1.0×L1 + 0.2×MS-SSIM`
  - L1: Pixel-wise reconstruction, weighted (masked regions 5× higher)
  - MS-SSIM: Multi-scale structural similarity (no external network required)
- **Why no adversarial loss:** GAN training adds instability and complexity. For 256×256 face inpainting, L1 + MS-SSIM achieves ~95-98% visual quality of GAN-based methods with simpler, more stable training.
- **Why no VGG perceptual loss:** VGG-19 adds ~140MB memory overhead. MS-SSIM captures perceptual quality across multiple scales without loading any pretrained network.
- **Mask Sources (train):** 50% CelebA semantic masks (nose, eyes, mouth) + 50% random rectangular masks
- **Random Mask Constraints:** Area <40% of image, random position and aspect ratio
- **Mask Sources (test):** CelebA-HQ test set (10k) with semantic masks → PSNR/SSIM metrics
- **Optimizer:** AdamW (lr=0.0002, weight_decay=0.01) with CosineAnnealingLR + linear warmup (1 epoch, start_lr=1e-5)
- **Epochs:** 3 (with early stopping, patience=2)
- **Validation:** Every epoch
- **Output:** `checkpoints/unet_best.pth`

**MS-SSIM Details:**
- Computes structural similarity at 5 scales (weights: 0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
- Captures both fine details (high freq) and global structure (low freq)
- Differentiable: direct backpropagation, no feature extraction network
- Range: [0, 1] where 1 = identical images

**M1 MPS Specific:**
- **Batch Size:** 8 (LiteUNet is 4× smaller than original)
- **Gradient Accumulation:** 2 steps (effective batch = 16)
- **Peak Memory:** ~3GB (vs ~6GB original)

**TensorBoard Logs:** Learning rate, gradient norm, PSNR, SSIM, per-loss components (L1, MS-SSIM)

---

### Step 4: Build the Search Index

**Script:** `python -m src.search.engine --model checkpoints/encoder_best.pth --output checkpoints/gallery_index.npz`

**Process (`src/search/engine.py`):** 80k images → Encoder → FAISS Index (64-dim embedding + 40-dim attributes per image)

**Attribute-Filtered Search:** `final_score = embedding_similarity × attribute_confidence`
- Embedding similarity: cosine similarity from FAISS (range [0, 1])
- Attribute confidence: sigmoid output for selected attribute (range [0, 1])
- Results re-ranked by final_score after initial top-K retrieval

Enables fast nearest-neighbor search with optional attribute filtering. Upgradeable to O(log n) with HNSW if gallery exceeds 500k images.

---

### Step 5: Run the Application

**Script:** `streamlit run src/app.py`

**The Streamlit app (`src/app.py`) provides 3 tabs:**

| Tab | Function | Flow |
|-----|----------|------|
| **Face Search** | Select query → Find similar faces | Query selection (Random/Filter/Upload) → Encoder → Search FAISS index → Top-K matches |
| **Inpainting** | Remove & reconstruct facial regions | Select region → Create mask → LiteUNet → Reconstructed image |
| **Grad-CAM** | Visualize what the encoder "sees" | Image → Encoder → Heatmap overlay |

**Search Options:**
- `☑ Include training images` — ON: search all 80k; OFF: search only test images + unused ones from the dataset
- `Attribute filters` — Re-rank by `similarity × attribute_confidence` for selected attributes

---

## Project Structure

```
CV_clone_search/
├── src/
│   ├── app.py                    # Streamlit GUI (main entry point)
│   ├── data/
│   │   ├── dataset.py            # CelebA & FFHQ data loaders
│   │   └── augmentations.py      # Albumentations pipelines
│   ├── models/
│   │   ├── encoder.py            # FaceEncoder (MobileNetV2 + embedding + attributes)
│   │   └── unet.py               # LiteUNet for inpainting
│   ├── training/
│   │   ├── train_encoder.py      # Encoder training script
│   │   └── train_unet.py         # LiteUNet training script
│   └── search/
│       └── engine.py             # FAISS search engine
├── configs/
│   └── config.yaml               # Hyperparameters
├── checkpoints/                  # Saved model weights
└── runs/                         # TensorBoard logs
```

**Removed from original:**
- `src/models/discriminator.py` — No adversarial training

---

## Key Technical Highlights

1. **Transfer Learning**: MobileNetV2 pretrained backbone with frozen early layers (~0.5M trainable params)
2. **SimCLR with Projection Head**: Contrastive loss on projection output protects embedding representation; projection discarded after training
3. **Large Effective Batch**: Gradient accumulation (8 steps) achieves effective batch of 256 for quality contrastive learning
4. **Calibrated Loss Weights**: Multi-task losses balanced by magnitude (0.5×SimCLR + 1.0×Attr)
5. **Class-Weighted Attribute Loss**: Handles CelebA class imbalance via positive/negative class weighting
6. **MS-SSIM Perceptual Loss**: Multi-scale structural similarity without external network (zero memory overhead)
7. **No Adversarial Training**: Simpler, stable training; L1 + MS-SSIM achieves ~95-98% visual quality of GANs
8. **LiteUNet (512 bottleneck)**: 4× fewer parameters than standard UNet (8M vs 31M)
9. **Compact Embeddings (64-dim)**: Balance between expressiveness and search speed
10. **FAISS Indexing**: O(n) exact search on 80k images, upgradeable to O(log n) with HNSW
11. **M1 MPS Optimized**: fits comfortably on 8GB RAM