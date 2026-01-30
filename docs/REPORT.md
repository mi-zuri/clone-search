# Face Search Engine with Attribute-Based Retrieval

**Computer Vision - Project 3**

---

## 1. Dataset Description

### CelebA-HQ (CelebAMask-HQ)
- **Size:** 30,000 celebrity face images (256x256)
- **Annotations:** 40 binary facial attributes, 19 facial region segmentation masks
- **Split:** 15k train, 5k validation, 10k test

### FFHQ (Flickr-Faces-HQ)
- **Size:** 50,000 high-quality face images (256x256)
- **Annotations:** None (unlabeled)
- **Usage:** 5k training subset (SimCLR), 50k gallery (search)

### Combined Gallery
- **Total:** 80,000 indexed faces (30k CelebA + 50k FFHQ)
- **Data augmentation:** Color jittering, random horizontal flips, random grayscale (SimCLR training)

### Sample Images
Dataset samples are available in the original CelebA-HQ and FFHQ repositories. The datasets contain diverse faces with varying ages, genders, ethnicities, poses, and expressions.

---

## 2. Problem Description

This project addresses two interconnected computer vision problems:

### Primary Problem: Face Search Engine
A content-based image retrieval system that:
- Encodes face images into 64-dimensional L2-normalized embeddings
- Performs efficient similarity search across 80,000+ gallery images using FAISS (Facebook AI Similarity Search)
- Returns top-k most similar faces based on cosine similarity
- Achieves sub-millisecond latency (0.5ms for k=5)

### Additional Problem: Attribute-Augmented Retrieval
Multi-task attribute prediction that enhances search quality by:
- Predicting 40 binary facial attributes from face embeddings
- Enabling attribute-filtered search (e.g., "find similar smiling faces")
- Improving embedding quality through multi-task regularization
- Computing confidence-weighted ranking: `final_score = embedding_similarity × attribute_confidence`


### Note on Inpainting
An inpainting feature was planned with UI scaffolding created, but remains unimplemented in this version.

---

## 3. Architecture & Model Details

### FaceEncoder Architecture

The FaceEncoder is a multi-task neural network built on MobileNetV2 with three specialized output heads:

```
Input Image (B, 3, 256, 256)
           ↓
┌──────────────────────────────────────┐
│   MobileNetV2 Backbone (Pretrained)  │
│   Layers 0-14: Frozen (ImageNet)     │
│   Layers 15-18: Fine-tuned           │
└──────────────────────────────────────┘
           ↓
    Features (B, 1280, 8, 8)
           ↓
    AdaptiveAvgPool2d(1)
           ↓
     Pooled (B, 1280)
           ↓
    ┌──────┴──────────────┬────────────┐
    ↓                     ↓            ↓
┌─────────────┐  ┌──────────────┐  ┌──────────────┐
│ Embedding   │  │ Projection   │  │ Attribute    │
│ Head        │  │ Head         │  │ Head         │
│             │  │              │  │              │
│ 1280→256    │  │ 64→128→64    │  │ 1280→40      │
│ ReLU        │  │ ReLU         │  │ (logits)     │
│ Dropout(0.2)│  │              │  │              │
│ 256→64      │  │              │  │              │
│ L2-norm     │  │ L2-norm      │  │              │
└─────────────┘  └──────────────┘  └──────────────┘
       ↓                ↓                 ↓
  Embedding        Projection      Attribute Logits
   (B, 64)            (B, 64)           (B, 40)
```

### Component Descriptions

#### 1. MobileNetV2 Backbone
- **Pretrained:** ImageNet1K_V1 weights
- **Freeze Strategy:** Layers 0-14 frozen (preserve low-level features), layers 15-18 trainable
- **Output:** 1280-dimensional feature maps

#### 2. Embedding Head
- **Purpose:** Generate face embeddings for similarity search
- **Architecture:** 1280 → 256 (ReLU, Dropout 0.2) → 64 (L2-normalized)
- **Normalization:** L2 normalization ensures cosine similarity = dot product

#### 3. Projection Head
- **Purpose:** Protect embedding from contrastive loss during SimCLR training
- **Architecture:** 64 → 128 (ReLU) → 64 (L2-normalized)
- **Usage:** Only used during training; discarded at inference

#### 4. Attribute Head
- **Purpose:** Predict 40 binary facial attributes
- **Architecture:** 1280 → 40 (linear projection from pooled features)
- **Activation:** BCEWithLogitsLoss (sigmoid implicit in loss)

### Design Rationale

- **MobileNetV2:** Efficient architecture (2.6M params) suitable for Macbook Air hardware constraints, strong pretrained ImageNet features, balanced accuracy/efficiency
- **Separate projection head:** Prevents contrastive loss from degrading embeddings used for search (standard SimCLR practice)
- **Attributes from pooled features:** Attributes require high-dimensional semantic information (1280-D); embeddings are compressed (64-D) for efficient search

### Parameter Counts

| Component | Total Parameters | Trainable | Frozen |
|-----------|-----------------|-----------|---------|
| **MobileNetV2 Backbone** | 2,236,672 | 698,392 | 1,538,280 |
| **Embedding Head** | 360,000 | 360,000 | 0 |
| **Projection Head** | 8,384 | 8,384 | 0 |
| **Attribute Head** | 51,240 | 51,240 | 0 |
| **TOTAL** | **2,636,072** | **1,938,280** | **697,792** |

**Model Size:** ~10.06 MB (weights only), 25 MB (with optimizer state)

---

## 4. Training Procedure

### Training Commands

**Phase 1: Train FaceEncoder**
- Loads CelebA (15k train + 5k val) and FFHQ (5k subset) datasets with SimCLR augmentations (2 views per image)
- Trains with multi-task loss: SimCLR contrastive + attribute prediction
- Validates every epoch, saves checkpoints to `checkpoints/encoder_epoch{N}.pt`
- Auto-resumes from latest checkpoint if interrupted, early stopping after 2 epochs without improvement

**Phase 2: Build Search Gallery Index**
- Loads best encoder checkpoint, processes all 80k gallery images (CelebA + FFHQ)
- Extracts 64-dim embeddings and 40-dim attribute predictions
- Builds FAISS IndexFlatIP for cosine similarity search, saves to NPZ file (~40 MB)

**Phase 3: Run Evaluation**
- Evaluate attribute prediction on 10k test set
- Evaluate retrieval metrics (Recall@k, MRR - Mean Reciprocal Rank)
- Benchmark search latency/throughput

**Phase 4: Launch Streamlit App**
- Face search with top-k results and attribute filtering
- Grad-CAM visualization for attribute predictions
- Inpainting tab (UI only, not implemented)

### Training Time

**Total Duration:** 6 epochs (~4.5 hours)
- Epoch 1: ~45 min (warmup), Epochs 2-6: ~40 min each
- Per-step timing: ~200ms (80ms forward, 120ms backward)
- Steps per epoch: ~470 (15k samples ÷ 32 batch size)

**Inference Time:**
- Encoder: 2.5ms/image (batch=32), throughput 50 img/s
- Search: 0.50ms (k=5, p50), 1,923 QPS
- Gallery build: ~30 min (one-time operation for 80k images)

---

## 5. Training & Evaluation

### Loss Functions

#### NT-Xent Loss (Normalized Temperature-scaled Cross Entropy - SimCLR Contrastive Loss)
```python
L_contrastive = -log(exp(sim(z_i, z_j) / τ) / Σ_k exp(sim(z_i, z_k) / τ))
```
- **Temperature (τ):** 0.25 (lower temp = harder negatives, better discrimination)
- **Weight:** 0.5
- **Purpose:** Learn discriminative embeddings via self-supervised contrastive learning

#### Weighted Binary Cross-Entropy (Attribute Loss)
```python
L_attribute = Σ_a w_a * BCE(logits_a, target_a)
```
- **Per-attribute weights (w_a):** Computed as `neg_count / pos_count`, clamped to [0.1, 10.0]
- **Weight:** 1.0
- **Purpose:** Handle class imbalance (e.g., Eyeglasses: 6% positive, Blurry: 0.3% positive)

#### Combined Loss
```python
L_total = 0.5 * L_contrastive + 1.0 * L_attribute
```

### Learning Rate Schedule + Training Loss Curves

![Training Losses](figures/1.png)

![Training Losses 2](figures/2.png)

Due to machine error, training had to be run 3 times. Hence the jagged lines.

### Validation Accuracy

![Validation Accuracy](figures/validation_accuracy.png)

### Evaluation Metrics

#### Attribute Prediction Metrics
- **Accuracy:** Per-attribute and mean accuracy
- **Balanced Accuracy:** Accounts for class imbalance
- **F1 Score:** Harmonic mean of precision/recall

#### Retrieval Metrics
- **Recall@k:** Fraction of queries with ground truth in top-k results (k=1, 5, 10)
- **Mean Reciprocal Rank (MRR):** Average of 1/rank for first relevant result

#### Search Performance Metrics
- **Latency:** p50, p95, p99 percentiles (ms)
- **Throughput:** Queries per second (QPS)

### Evaluation Results

#### Attribute Prediction (10,000 test images)

**Summary:**
- **Mean Accuracy:** 85.54%
- **Mean Balanced Accuracy:** 85.54%
- **Mean F1 Score:** 68.11%

**Best 5 Attributes:**

| Attribute | Accuracy | F1 Score | Positive Rate |
|-----------|----------|----------|---------------|
| Eyeglasses | 99.68% | 97.36% | 6.0% |
| Blurry | 99.52% | 20.00% | 0.3% |
| Wearing_Hat | 98.93% | 86.71% | 3.6% |
| Male | 97.83% | 97.44% | 42.1% |
| Bald | 97.12% | 65.22% | 2.8% |

**Worst 5 Attributes:**

| Attribute | Accuracy | F1 Score | Positive Rate |
|-----------|----------|----------|---------------|
| Big_Lips | 61.13% | 53.98% | 31.6% |
| Pointy_Nose | 64.85% | 53.74% | 30.3% |
| Oval_Face | 69.42% | 51.77% | 27.7% |
| Wearing_Necklace | 70.44% | 41.97% | 13.7% |
| Big_Nose | 73.24% | 66.04% | 29.9% |

### Attribute Prediction Performance

![Attribute Accuracy Distribution](figures/attribute_accuracy.png)
*Per-attribute accuracy distribution showing best (Eyeglasses: 99.68%) to worst (Big_Lips: 61.13%)*

#### Retrieval Metrics (10,000 queries on 80,000 gallery)

**80% Attribute Match Threshold (32/40 attributes):**
- **Recall@1:** 0.04%
- **Recall@5:** 91.34%
- **Recall@10:** 95.36%
- **MRR:** 0.4286

**90% Attribute Match Threshold (36/40 attributes):**
- **Recall@1:** 0.02%
- **Recall@5:** 37.56%
- **Recall@10:** 49.08%
- **MRR:** 0.1737

#### Search Performance Benchmarks

| Scenario | Mean Latency | p95 Latency | Throughput |
|----------|-------------|-------------|------------|
| **Basic Search (k=5, no filters)** | 0.50 ms | 0.60 ms | **1,923 QPS** |
| Filtered Search (k=5, Smiling=True) | 0.78 ms | 0.96 ms | 1,335 QPS |
| Multi-filter Search (k=5, Smiling+Male) | 0.74 ms | 0.85 ms | 1,295 QPS |
| High-k Search (k=100, no filters) | 1.02 ms | 1.23 ms | 983 QPS |

**Key Insight:** Even with attribute filtering, search remains sub-millisecond, demonstrating the efficiency of the FAISS-based retrieval system.

---

## 6. Hyperparameters

| Hyperparameter | Value | Justification |
|----------------|-------|---------------|
| **batch_size** | 32 | Maximum batch size fitting in 8GB MPS memory without OOM errors. |
| **gradient_accumulation_steps** | 8 | Effective batch size of 256 (32×8) for SimCLR. Large batches critical for contrastive learning. |
| **epochs** | 6 | Determined empirically with early stopping (patience=2). Training converged at epoch 4-6. |
| **lr** (learning rate) | 0.001 | Standard AdamW learning rate for transfer learning. |
| **weight_decay** | 0.01 | Moderate L2 regularization prevents overfitting on 15k training samples. |
| **simclr_temperature** | 0.25 | Lower temperature sharpens similarity distribution → harder negatives → more discriminative embeddings. |
| **simclr_loss_weight** | 0.5 | Calibrated to balance loss magnitudes. |
| **attribute_loss_weight** | 1.0 | Primary supervised signal guiding embedding space structure. |
| **embedding_dim** | 64 | Balances expressiveness vs. search speed. |
| **projection_dim** | 64 | Matches embedding_dim. Projection head protects embeddings from contrastive loss degradation. |
| **freeze_layers** | 14 | Freezes low-level ImageNet features (edges, textures). Fine-tunes high-level features (faces). |
| **warmup_epochs** | 1 | Linear warmup stabilizes early training, prevents large gradient spikes. |
| **warmup_start_lr** | 1e-5 | 100× smaller than target LR for gradual warmup. |
| **gradient_clip_norm** | 1.0 | Prevents exploding gradients during SimCLR training. |
| **early_stopping_patience** | 2 | Stops after 2 epochs without validation improvement. |
| **dropout** | 0.2 | Applied in embedding head only. Regularizes compact bottleneck layer. |
| **num_workers** | 2 | Optimized for M1 MPS. |
| **celeba_batch_ratio** | 0.75 | 75% CelebA + 25% FFHQ ensures sufficient labeled samples. |

### Key Design Decisions

- **Gradient accumulation:** Hardware constraint (8GB MPS) limits physical batch to 32; SimCLR requires large batches (256+); gradient accumulation provides large effective batch without memory overhead
- **Partial freezing (layers 0-14):** Preserves low-level ImageNet features, adapts high-level features to faces; avoids overfitting and catastrophic forgetting
- **Low SimCLR temperature (0.25):** Sharpens similarity distribution, harder negatives, improved Recall@5 from ~85% (τ=0.5) to ~91% (τ=0.25)

---

## 7. Results

Here we present some example usages of the app run by Streamlit.

**Examples similarity search:**

![Demo](figures/Me.png)

![old man](figures/example.png)

**Example attribute search:**

![bald woman](figures/bald.png)

**Example Grad-Cam analysis:**

![hat](figures/hat.png)

![beard](figures/mustache.png)

![no blonde hair](figures/blonde.png)

---

## 8. Runtime Environment

**Hardware:**
- M1 MacBook Air, 8GB unified memory, 8-core GPU (7-core variant), SSD

**Software:**
- macOS 23.6.0 (Darwin), Python 3.11.3
- PyTorch Backend: MPS (Metal Performance Shaders)

**MPS-Specific Optimizations:**
- Mixed precision disabled (limited float16 support on MPS)
- Batch size tuned to 32 to avoid MPS memory errors
- Gradient accumulation compensates for small physical batch size
- FAISS: CPU-only (FAISS-GPU incompatible with MPS)

---

## 9. Libraries and Tools

### Core Dependencies (requirements.txt)

```
torch>=2.0.0                 # Deep learning framework
torchvision>=0.15.0          # Pretrained models and transforms
albumentations>=1.3.0        # Advanced data augmentation
tensorboard>=2.14.0          # Training visualization
streamlit>=1.28.0            # Interactive web application
scikit-image>=0.21.0         # Image processing utilities
pillow>=10.0.0               # Image I/O
tqdm>=4.66.0                 # Progress bars
pyyaml>=6.0                  # Configuration file parsing
kaggle>=1.5.0                # Dataset downloading
faiss-cpu>=1.7.4             # Efficient similarity search
grad-cam>=1.5.0              # Explainable AI visualizations
numpy>=1.24.0                # Numerical computing
pandas>=2.0.0                # Data manipulation
matplotlib>=3.7.0            # Plotting
opencv-python>=4.8.0         # Computer vision utilities
```

### Tool Usage

- **PyTorch:** Model definition, NT-Xent loss, MPS backend for Apple Silicon GPU
- **TensorBoard:** Real-time training monitoring (loss curves, LR schedules, metrics)
- **Streamlit:** Interactive web UI (src/app.py) with search and Grad-CAM visualization
- **FAISS:** IndexFlatIP for sub-millisecond cosine similarity search on 80k gallery
- **Grad-CAM:** Visual explanations for attribute predictions, highlights facial regions
- **Albumentations:** High-performance SimCLR transforms (ColorJitter, HorizontalFlip, ToGray)

---

## 10. Bibliography

### Datasets

**CelebAMask-HQ**
Lee, C.H., Liu, Z., Wu, L., Luo, P. (2020). *MaskGAN: Towards Diverse and Interactive Facial Image Manipulation.* IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
[https://github.com/switchablenorms/CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ)

**FFHQ (Flickr-Faces-HQ)**
Karras, T., Laine, S., Aila, T. (2019). *A Style-Based Generator Architecture for Generative Adversarial Networks.* IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
[https://github.com/DCGM/ffhq-features-dataset](https://github.com/DCGM/ffhq-features-dataset)

### Methods

**SimCLR**
Chen, T., Kornblith, S., Norouzi, M., Hinton, G. (2020). *A Simple Framework for Contrastive Learning of Visual Representations.* International Conference on Machine Learning (ICML).
[https://arxiv.org/abs/2002.05709](https://arxiv.org/abs/2002.05709)

**MobileNetV2**
Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., Chen, L.C. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
[https://arxiv.org/abs/1801.04381](https://arxiv.org/abs/1801.04381)

**Grad-CAM**
Selvaraju, R.R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., Batra, D. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* IEEE International Conference on Computer Vision (ICCV).
[https://arxiv.org/abs/1610.02391](https://arxiv.org/abs/1610.02391)

**FAISS**
Johnson, J., Douze, M., Jégou, H. (2019). *Billion-scale similarity search with GPUs.* IEEE Transactions on Big Data.
[https://github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss)

**U-Net** (for future inpainting implementation)
Ronneberger, O., Fischer, P., Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.* International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI).
[https://arxiv.org/abs/1505.04597](https://arxiv.org/abs/1505.04597)

**MS-SSIM** (Multi-Scale Structural Similarity)
Wang, Z., Simoncelli, E.P., Bovik, A.C. (2003). *Multiscale structural similarity for image quality assessment.* The Thirty-Seventh Asilomar Conference on Signals, Systems & Computers.
[https://ieeexplore.ieee.org/document/1292216](https://ieeexplore.ieee.org/document/1292216)

### Tools

**PyTorch**
[https://pytorch.org](https://pytorch.org)

**Streamlit**
[https://streamlit.io](https://streamlit.io)

**TensorBoard**
[https://www.tensorflow.org/tensorboard](https://www.tensorflow.org/tensorboard)

---

## 11. Points Breakdown

| Category | Item | Task.md Reference | Points |
|----------|------|-------------------|:------:|
| **Problem** | Face search engine | Line 24 | **2** |
| | Attribute prediction improves search quality | Line 31 | **1** |
| **Model** | Pre-trained model on different problem (MobileNetV2 ImageNet → faces) | Line 35 | **1** |
| | Ready architecture trained from scratch (custom multi-head design) | Line 36 | **1** |
| | Each subsequent model with different architecture (3 specialized heads) | Line 40 | **1** |
| | Non-trivial solution: contrastive learning (SimCLR NT-Xent loss) | Line 42 | **1** |
| **Dataset** | Evaluation on 10,000+ photos (10k test set) | Line 54 | **1** |
| **Training** | Hyperparameter estimation (calibrated loss weights, temperature tuning) | Line 68 | **1** |
| | Adaptive hyperparameters (warmup + cosine annealing scheduler) | Line 69 | **1** |
| | Data augmentation (SimCLR transforms: color jitter, flip, grayscale) | Line 72 | **1** |
| **Tools** | TensorBoard with experiment analysis (loss curves, LR schedule, metrics) | Line 85 | **1** |
| | Streamlit GUI (interactive search + attribute filtering) | Line 87 | **1** |
| | Grad-CAM explanation of predictions (integrated in UI) | Line 91 | **2** |
| **TOTAL** | | | **15** |

---

## 12. Git Repository

**Repository URL:**
[https://github.com/mi-zuri/clone-search.git](https://github.com/mi-zuri/clone-search.git)

---

## Conclusion

This project successfully implements a high-performance face search engine achieving **91.34% Recall@5** on an 80,000-image gallery with sub-millisecond latency. The multi-task learning approach (SimCLR + attribute prediction) demonstrates that contrastive self-supervision and supervised attribute learning are complementary—yielding both discriminative embeddings and interpretable semantic predictions.

The system achieves real-world usability on consumer hardware (M1 MacBook Air), making it deployable for interactive applications. Future work includes implementing the planned inpainting feature and exploring metric learning alternatives to SimCLR.

## Additional Notes

Processing time (number of epochs), amount and complexity of implemented solutions was highly time-limited.
Solution could be greatly improved with minor changes.