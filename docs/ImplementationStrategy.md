# CV Clone Search - Implementation Strategy

A phased implementation plan for the multi-task face analysis system.

---

## Executive Summary

**Goal:** Build a face analysis system with three capabilities:
1. **Face Search** - Find similar faces using learned embeddings
2. **Face Inpainting** - Reconstruct missing facial regions (deferred)
3. **Attribute Prediction** - Classify 40 binary facial attributes

**Models:** FaceEncoder (MobileNetV2 → 64-dim embeddings) + LiteUNet (~8M params)

**Target Hardware:** M1 MacBook Air (8GB), MPS backend

**Datasets:** CelebA-HQ (30k) + FFHQ (50k) = 80k images

---

## Phase Overview

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | ✅ Complete | Project Foundation & Data Pipeline |
| 2 | ✅ Complete | FaceEncoder Architecture |
| 3 | ✅ Complete | Encoder Training (6 epochs) |
| 4 | 🔄 Next | Search Engine & Indexing |
| 5 | ⏳ Pending | Streamlit App (with inpainting stub) |
| 6 | ⏳ Pending | Evaluation - Encoder & Search |
| 7 | ⏳ Deferred | LiteUNet Architecture & Training |
| 8 | ⏳ Deferred | Inpainting Integration & Full Eval |

---

## Phase 1: Project Foundation & Data Pipeline ✅

### 1.1 Project Structure Setup

Create the directory structure:

```
CV_clone_search/
├── src/
│   ├── __init__.py
│   ├── app.py                    # Streamlit GUI (Phase 5)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py            # Data loaders
│   │   └── augmentations.py      # Albumentations pipelines
│   ├── models/
│   │   ├── __init__.py
│   │   ├── encoder.py            # FaceEncoder
│   │   └── unet.py               # LiteUNet
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_encoder.py      # Encoder training
│   │   └── train_unet.py         # UNet training
│   └── search/
│       ├── __init__.py
│       └── engine.py             # FAISS search
├── configs/
│   └── config.yaml               # All hyperparameters
├── checkpoints/                  # Model weights
├── runs/                         # TensorBoard logs
└── data/                         # Datasets (not in repo)
```

### 1.2 Configuration File (`configs/config.yaml`)

```yaml
# Data
data:
  celeba_dir: "data/celeba"
  celeba_masks_dir: "data/celeba_masks"
  ffhq_dir: "data/ffhq"
  attributes_file: "data/CelebAMask-HQ-attribute-anno.txt"
  image_size: 256

# Data splits
splits:
  celeba_train: 15000      # Supervised training (attributes)
  celeba_val: 5000
  celeba_test: 10000
  ffhq_simclr_subset: 5000 # Random subset for SimCLR

# Encoder training
encoder:
  batch_size: 32
  grad_accumulation: 8     # Effective batch = 256
  epochs: 5
  lr: 0.001
  warmup_epochs: 1
  warmup_start_lr: 1e-5
  weight_decay: 0.01
  grad_clip: 1.0
  early_stopping_patience: 2

  # Loss weights
  simclr_weight: 0.5
  attribute_weight: 1.0
  simclr_temperature: 0.25

  # Architecture
  embedding_dim: 64
  projection_dim: 64
  freeze_layers: 14        # MobileNetV2 layers 0-14 frozen

# UNet training (Phase 7)
unet:
  batch_size: 8
  grad_accumulation: 2     # Effective batch = 16
  epochs: 3
  lr: 0.0002
  warmup_epochs: 1
  warmup_start_lr: 1e-5
  weight_decay: 0.01
  early_stopping_patience: 2

  # Loss weights
  l1_weight: 1.0
  msssim_weight: 0.2
  masked_region_weight: 5.0

  # Architecture
  base_channels: 32        # Half of standard UNet

  # Masking
  semantic_mask_ratio: 0.5 # 50% semantic, 50% random
  max_random_mask_area: 0.4

# MPS settings
mps:
  num_workers: 2
  use_float32: true        # No mixed precision on MPS
```

### 1.3 Data Loaders (`src/data/dataset.py`)

**Expected folder structure:**
```
data/
├── celeba/                              # 30k images
│   ├── 0/ ... 30/                       # 31 subfolders
│   │   └── {id}.jpg                     # 0.jpg, 1.jpg, ...
├── celeba_masks/                        # Semantic masks
│   ├── 0/ ... 372/
│   │   └── {id:05d}_{region}.png        # 00000_nose.png
├── ffhq/                                # 50k images
│   ├── 0/ ... 52/                       # 53 subfolders
│   │   └── {id:05d}.png                 # 00000.png
└── CelebAMask-HQ-attribute-anno.txt     # Attributes
```

**Mask regions (18 types):** `cloth`, `ear_r`, `eye_g`, `hair`, `hat`, `l_brow`, `l_ear`, `l_eye`, `l_lip`, `mouth`, `neck`, `neck_l`, `nose`, `r_brow`, `r_ear`, `r_eye`, `skin`, `u_lip`

**Attribute file format:**
- Line 1: `30000` (image count)
- Line 2: 40 attribute names (space-separated)
- Lines 3+: `{filename}  {40 × (-1|1)}` where -1=absent, 1=present

**Implementation tasks:**

1. **`CelebADataset`** - Loads CelebA images + attributes + optional masks
   - Parse annotation file (handle double-space delimiter)
   - Convert -1/1 → 0/1 for BCEWithLogits
   - Load masks by region name (nose, l_eye, r_eye, mouth)

2. **`FFHQDataset`** - Loads FFHQ images (no labels)
   - Support random subset sampling for SimCLR training

3. **`CombinedDataset`** - Merges both for training
   - 75% CelebA + 25% FFHQ per batch
   - Return `has_attributes` flag per sample

4. **`SimCLRWrapper`** - Returns two augmented views per image
   - Applied to any base dataset

### 1.4 Augmentations (`src/data/augmentations.py`)

**SimCLR augmentations (training):**
```python
A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
    A.GaussianBlur(blur_limit=(3, 7), p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])
```

**Standard augmentations (UNet training):**
```python
A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])
```

**Validation/Test:**
```python
A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])
```

### Phase 1 Deliverables ✅

- [x] Project structure created
- [x] `config.yaml` with all hyperparameters
- [x] `CelebADataset` with attribute parsing and mask loading
- [x] `FFHQDataset` with subset sampling
- [x] `CombinedDataset` with 75/25 mixing
- [x] `SimCLRWrapper` for contrastive pairs
- [x] Augmentation pipelines
- [x] Unit tests for data loading

---

## Phase 2: FaceEncoder Architecture ✅

### 2.1 Model Architecture (`src/models/encoder.py`)

```
Input: (B, 3, 256, 256) RGB image
          ↓
┌─────────────────────────────────────────┐
│  MobileNetV2 Backbone (pretrained)       │
│  ├── Layers 0-14: FROZEN                │
│  └── Layers 15-18: TRAINABLE            │
└─────────────────────────────────────────┘
          ↓
    Global Average Pooling
          ↓
    (B, 1280) feature vector
          ↓
    ┌─────┴─────┐
    ↓           ↓
┌───────┐  ┌────────────┐
│Embed  │  │Attribute   │
│Head   │  │Head        │
└───────┘  └────────────┘
    ↓           ↓
┌───────┐  ┌────────────┐
│(B,64) │  │(B, 40)     │
│L2-norm│  │logits      │
└───────┘  └────────────┘
    ↓
┌───────────┐
│Projection │ (training only)
│Head       │
└───────────┘
    ↓
  (B, 64) L2-normalized
```

**Embedding Head:**
```python
nn.Sequential(
    nn.Linear(1280, 256),
    nn.ReLU(inplace=True),
    nn.Dropout(0.2),
    nn.Linear(256, 64),
)
# + L2 normalization
```

**Projection Head (SimCLR, discarded after training):**
```python
nn.Sequential(
    nn.Linear(64, 128),
    nn.ReLU(inplace=True),
    nn.Linear(128, 64),
)
# + L2 normalization
```

**Attribute Head:**
```python
nn.Linear(1280, 40)  # Direct from backbone features
```

### 2.2 Implementation Details

```python
class FaceEncoder(nn.Module):
    def __init__(self, embedding_dim=64, projection_dim=64, freeze_layers=14):
        # Load pretrained MobileNetV2
        backbone = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)

        # Remove classifier, keep features
        self.features = backbone.features  # 19 blocks (0-18)

        # Freeze layers 0-14
        for i, layer in enumerate(self.features):
            if i <= freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False

        # Heads
        self.embedding_head = ...
        self.projection_head = ...
        self.attribute_head = ...

    def forward(self, x, return_projection=False):
        # Backbone
        features = self.features(x)  # (B, 1280, 8, 8)
        pooled = F.adaptive_avg_pool2d(features, 1).flatten(1)  # (B, 1280)

        # Embedding (always computed)
        embedding = self.embedding_head(pooled)
        embedding = F.normalize(embedding, p=2, dim=1)

        # Attributes (always computed)
        attributes = self.attribute_head(pooled)

        if return_projection:
            projection = self.projection_head(embedding)
            projection = F.normalize(projection, p=2, dim=1)
            return embedding, projection, attributes

        return embedding, attributes
```

### 2.3 Trainable Parameters

| Component | Parameters | Trainable |
|-----------|------------|-----------|
| MobileNetV2 layers 0-14 | ~2.2M | No |
| MobileNetV2 layers 15-18 | ~0.5M | Yes |
| Embedding head | ~0.35M | Yes |
| Projection head | ~0.02M | Yes (discarded) |
| Attribute head | ~0.05M | Yes |
| **Total trainable** | **~0.9M** | |

### Phase 2 Deliverables ✅

- [x] `FaceEncoder` class with MobileNetV2 backbone
- [x] Proper layer freezing (0-14 frozen)
- [x] Embedding head with L2 normalization
- [x] Projection head for SimCLR
- [x] Attribute head for 40 binary attributes
- [x] Forward method with optional projection output
- [x] Unit tests for output shapes

---

## Phase 3: Encoder Training Pipeline ✅

### 3.1 Loss Functions

**NT-Xent Loss (SimCLR):**
```python
def nt_xent_loss(z1, z2, temperature=0.25):
    """
    z1, z2: (B, D) L2-normalized projection vectors
    Returns scalar loss
    """
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)  # (2B, D)

    # Cosine similarity matrix
    sim = torch.mm(z, z.T) / temperature  # (2B, 2B)

    # Mask out self-similarity
    mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim.masked_fill_(mask, -float('inf'))

    # Positive pairs: (i, i+B) and (i+B, i)
    labels = torch.cat([torch.arange(B, 2*B), torch.arange(B)], dim=0).to(z.device)

    loss = F.cross_entropy(sim, labels)
    return loss
```

**Attribute Loss (Class-Weighted BCE):**
```python
def attribute_loss(logits, targets, pos_weights):
    """
    logits: (B, 40) raw logits
    targets: (B, 40) binary labels (0 or 1)
    pos_weights: (40,) weights for positive class
    """
    return F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weights
    )
```

**Computing positive weights from training set:**
```python
# For each attribute, compute: neg_count / pos_count
# This upweights rare positives (e.g., Bald ~2% → weight ~50)
pos_weights = (1 - attr_means) / attr_means
pos_weights = torch.clamp(pos_weights, min=0.1, max=10.0)  # Stability
```

### 3.2 Training Loop (`src/training/train_encoder.py`)

```python
# Training loop pseudocode
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(train_loader):
        # Unpack batch
        img1, img2 = batch['view1'], batch['view2']  # SimCLR pairs
        attributes = batch['attributes']              # (B, 40) or None
        has_attrs = batch['has_attributes']           # (B,) bool mask

        # Forward pass
        emb1, proj1, attr_logits1 = model(img1, return_projection=True)
        emb2, proj2, _ = model(img2, return_projection=True)

        # SimCLR loss (all samples)
        loss_simclr = nt_xent_loss(proj1, proj2, temperature=0.25)

        # Attribute loss (CelebA samples only)
        if has_attrs.any():
            mask = has_attrs
            loss_attr = attribute_loss(
                attr_logits1[mask],
                attributes[mask],
                pos_weights
            )
        else:
            loss_attr = 0.0

        # Combined loss
        loss = 0.5 * loss_simclr + 1.0 * loss_attr
        loss = loss / grad_accumulation_steps
        loss.backward()

        # Gradient accumulation
        if (batch_idx + 1) % grad_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

    # Validation
    val_metrics = validate(model, val_loader)

    # Early stopping
    if val_metrics['mean_acc'] > best_acc:
        best_acc = val_metrics['mean_acc']
        save_checkpoint(model, 'encoder_best.pth')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break
```

### 3.3 Learning Rate Schedule

```python
# Warmup + Cosine Annealing
def get_lr(epoch, batch_idx, total_batches, config):
    warmup_steps = config.warmup_epochs * total_batches
    total_steps = config.epochs * total_batches
    current_step = epoch * total_batches + batch_idx

    if current_step < warmup_steps:
        # Linear warmup
        return config.warmup_start_lr + (config.lr - config.warmup_start_lr) * (current_step / warmup_steps)
    else:
        # Cosine annealing
        progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
        return config.lr * 0.5 * (1 + math.cos(math.pi * progress))
```

### 3.4 Validation Metrics

```python
def validate(model, val_loader):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in val_loader:
            _, attr_logits = model(batch['image'])
            preds = (torch.sigmoid(attr_logits) > 0.5).float()
            all_preds.append(preds)
            all_targets.append(batch['attributes'])

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    # Per-attribute accuracy
    per_attr_acc = (all_preds == all_targets).float().mean(dim=0)

    return {
        'mean_acc': per_attr_acc.mean().item(),
        'per_attr_acc': per_attr_acc.cpu().numpy(),
    }
```

### 3.5 TensorBoard Logging

Log every N batches:
- `train/loss_total`
- `train/loss_simclr`
- `train/loss_attribute`
- `train/learning_rate`
- `train/grad_norm`
- `train/embedding_l2_norm` (mean, std)

Log every epoch:
- `val/mean_accuracy`
- `val/attr_{name}_accuracy` (40 attributes)

### 3.6 MPS-Specific Considerations

```python
# Device setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# DataLoader settings
train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2,
    pin_memory=False,  # Not needed for MPS
    persistent_workers=True,
    multiprocessing_context='spawn',  # Required for macOS
)

# No mixed precision (MPS has limited float16 support)
# Use float32 throughout
```

### Phase 3 Deliverables ✅

- [x] NT-Xent loss implementation
- [x] Class-weighted BCE loss for attributes
- [x] Compute positive weights from training data
- [x] Training loop with gradient accumulation
- [x] Warmup + cosine annealing scheduler
- [x] Validation function with per-attribute accuracy
- [x] TensorBoard logging
- [x] Early stopping with patience=2
- [x] Checkpoint saving/loading
- [x] CLI with argparse + config loading

**Run command:** `python -m src.training.train_encoder --config configs/config.yaml`

**Training completed:** 6 epochs, checkpoints saved to `checkpoints/`

---

## Phase 4: Search Engine & Indexing 🔄

### 4.1 Gallery Building (`src/search/engine.py`)

**Key decisions:**
- Use `encoder_best.pth` (best validation accuracy)
- Include both CelebA (30k) + FFHQ (50k) = 80k images

```python
def build_gallery(encoder, celeba_dataset, ffhq_dataset, output_path):
    """
    Process all 80k images and save:
    - embeddings: (80000, 64) float32
    - attributes: (80000, 40) float32 (sigmoid outputs)
    - image_paths: list of 80000 paths
    - sources: list of 80000 'celeba'/'ffhq' labels
    """
    encoder.eval()
    embeddings, attributes, paths, sources = [], [], [], []

    # Process CelebA (has attributes)
    with torch.no_grad():
        for batch in DataLoader(celeba_dataset, batch_size=64):
            emb, attr_logits = encoder(batch['image'].to(device))
            attr_probs = torch.sigmoid(attr_logits)

            embeddings.append(emb.cpu().numpy())
            attributes.append(attr_probs.cpu().numpy())
            paths.extend(batch['path'])
            sources.extend(['celeba'] * len(batch['path']))

    # Process FFHQ (no ground truth attributes, but predict anyway)
    with torch.no_grad():
        for batch in DataLoader(ffhq_dataset, batch_size=64):
            emb, attr_logits = encoder(batch['image'].to(device))
            attr_probs = torch.sigmoid(attr_logits)

            embeddings.append(emb.cpu().numpy())
            attributes.append(attr_probs.cpu().numpy())
            paths.extend(batch['path'])
            sources.extend(['ffhq'] * len(batch['path']))

    embeddings = np.vstack(embeddings)      # (80000, 64)
    attributes = np.vstack(attributes)      # (80000, 40)

    np.savez(output_path,
        embeddings=embeddings,
        attributes=attributes,
        paths=np.array(paths),
        sources=np.array(sources),
    )
```

### 4.2 FAISS Index

```python
import faiss

class FaceSearchEngine:
    def __init__(self, gallery_path):
        data = np.load(gallery_path, allow_pickle=True)
        self.embeddings = data['embeddings'].astype('float32')
        self.attributes = data['attributes']
        self.paths = data['paths']
        self.sources = data['sources']

        # Build FAISS index (exact search with cosine similarity)
        # Normalize embeddings for inner product = cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index = faiss.IndexFlatIP(64)  # Inner product on L2-normalized = cosine
        self.index.add(self.embeddings)

    def search(self, query_embedding, k=10, attribute_filters=None):
        """
        query_embedding: (64,) L2-normalized
        attribute_filters: dict {attr_idx: min_confidence} or None
        Returns: list of (path, similarity, attr_confidences)
        """
        # L2 normalize query
        query = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query)

        # Initial search (get more results for re-ranking)
        k_initial = k * 5 if attribute_filters else k
        similarities, indices = self.index.search(query, k_initial)

        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            path = self.paths[idx]
            attr_conf = self.attributes[idx]
            source = self.sources[idx]

            # Attribute filtering
            if attribute_filters:
                final_score = sim
                for attr_idx, min_conf in attribute_filters.items():
                    final_score *= attr_conf[attr_idx]
            else:
                final_score = sim

            results.append({
                'path': path,
                'similarity': sim,
                'final_score': final_score,
                'attributes': attr_conf,
                'source': source,
            })

        # Re-rank by final score
        results.sort(key=lambda x: x['final_score'], reverse=True)
        return results[:k]
```

### 4.3 CLI for Index Building

```bash
python -m src.search.engine \
    --encoder checkpoints/encoder_best.pth \
    --output checkpoints/gallery_index.npz
```

### Phase 4 Deliverables

- [ ] Gallery building function (process 80k images: 30k CelebA + 50k FFHQ)
- [ ] Save embeddings + attributes + paths + sources to NPZ
- [ ] FAISS index with cosine similarity (IndexFlatIP)
- [ ] Search function with optional attribute filtering
- [ ] Re-ranking by `similarity × attribute_confidence`
- [ ] CLI for building index

---

## Phase 5: Streamlit Application

### 5.1 App Structure (`src/app.py`)

```python
import streamlit as st

st.set_page_config(page_title="CV Clone Search", layout="wide")

# Sidebar: Model loading
@st.cache_resource
def load_models():
    encoder = FaceEncoder()
    encoder.load_state_dict(torch.load('checkpoints/encoder_best.pth'))
    encoder.eval()

    search_engine = FaceSearchEngine('checkpoints/gallery_index.npz')

    return encoder, search_engine

encoder, search_engine = load_models()

# Tabs
tab1, tab2, tab3 = st.tabs(["Face Search", "Inpainting", "Grad-CAM"])
```

### 5.2 Tab 1: Face Search

```python
with tab1:
    st.header("Face Search")

    # Query selection
    query_method = st.radio("Select query image:",
                           ["Random from gallery", "Filter by attributes", "Upload"])

    if query_method == "Random from gallery":
        if st.button("🎲 Random"):
            query_path = np.random.choice(search_engine.paths)
            st.session_state.query_path = query_path

    elif query_method == "Filter by attributes":
        # Attribute checkboxes
        selected_attrs = st.multiselect("Filter by:", ATTRIBUTE_NAMES)
        # ... filter and select

    elif query_method == "Upload":
        uploaded = st.file_uploader("Upload image", type=['jpg', 'png'])
        if uploaded:
            # Save to temp, process
            pass

    # Search options
    col1, col2 = st.columns(2)
    with col1:
        include_train = st.checkbox("Include training images", value=True)
    with col2:
        k = st.slider("Number of results", 5, 50, 10)

    # Attribute filters for search
    attr_filters = st.multiselect("Re-rank by attributes:", ATTRIBUTE_NAMES)

    # Run search
    if st.session_state.get('query_path'):
        query_img = load_image(st.session_state.query_path)
        query_emb, _ = encoder(query_img.unsqueeze(0))

        filter_dict = {ATTR_TO_IDX[a]: 0.5 for a in attr_filters} if attr_filters else None
        results = search_engine.search(query_emb[0], k=k, attribute_filters=filter_dict)

        # Display results
        st.image(query_img, caption="Query", width=200)

        cols = st.columns(5)
        for i, res in enumerate(results):
            with cols[i % 5]:
                st.image(res['path'], caption=f"Sim: {res['similarity']:.3f}")
```

### 5.3 Tab 2: Inpainting (Stub - Coming Soon)

```python
with tab2:
    st.header("Face Inpainting")

    st.info("🚧 **Coming Soon** - UNet model not yet trained. This feature will be available after Phase 7.")

    st.markdown("""
    **Planned features:**
    - Upload face image
    - Select region to inpaint (Nose, Eyes, Mouth, Custom Rectangle)
    - AI-powered facial region reconstruction
    """)

    # Keep UI skeleton for preview
    uploaded = st.file_uploader("Upload face image", type=['jpg', 'png'], key='inpaint', disabled=True)

    region = st.selectbox("Select region to inpaint:",
                         ["Nose", "Left Eye", "Right Eye", "Mouth", "Custom Rectangle"],
                         disabled=True)

    st.button("Inpaint", disabled=True)
```

### 5.4 Tab 3: Grad-CAM Visualization

```python
with tab3:
    st.header("Grad-CAM Visualization")

    uploaded = st.file_uploader("Upload face image", type=['jpg', 'png'], key='gradcam')

    if uploaded:
        image = load_image(uploaded)

        # Target selection
        target = st.selectbox("Visualize attention for:",
                             ["Embedding (overall face)", *ATTRIBUTE_NAMES])

        if st.button("Generate Grad-CAM"):
            # Compute Grad-CAM
            heatmap = compute_gradcam(encoder, image, target)

            # Overlay on image
            overlay = apply_heatmap(image, heatmap)

            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original")
            with col2:
                st.image(overlay, caption=f"Grad-CAM: {target}")
```

### 5.5 Grad-CAM Implementation

```python
def compute_gradcam(model, image, target_attr_idx=None):
    """
    Compute Grad-CAM for embedding or specific attribute.
    """
    model.eval()

    # Hook to capture gradients and activations
    gradients = []
    activations = []

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    def forward_hook(module, input, output):
        activations.append(output)

    # Register hooks on last conv layer
    target_layer = model.features[-1]  # Last MobileNetV2 block
    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    # Forward pass
    image.requires_grad = True
    emb, attr_logits = model(image.unsqueeze(0))

    # Backward pass
    if target_attr_idx is not None:
        target = attr_logits[0, target_attr_idx]
    else:
        target = emb.sum()  # Embedding magnitude

    model.zero_grad()
    target.backward()

    # Compute Grad-CAM
    grads = gradients[0]              # (1, C, H, W)
    acts = activations[0]             # (1, C, H, W)

    weights = grads.mean(dim=[2, 3])  # (1, C)
    cam = (weights.unsqueeze(-1).unsqueeze(-1) * acts).sum(dim=1)  # (1, H, W)
    cam = F.relu(cam)
    cam = F.interpolate(cam.unsqueeze(1), size=(256, 256), mode='bilinear')
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    # Cleanup
    fh.remove()
    bh.remove()

    return cam[0, 0].detach().cpu().numpy()
```

### Phase 5 Deliverables

- [ ] Streamlit app with 3 tabs
- [ ] Face Search tab with query options (fully functional)
- [ ] Attribute filtering and re-ranking
- [ ] Inpainting tab with "Coming Soon" stub
- [ ] Grad-CAM visualization tab (fully functional)
- [ ] Model caching with `@st.cache_resource`
- [ ] Responsive layout

**Run command:** `streamlit run src/app.py`

---

## Phase 6: Evaluation - Encoder & Search

### 6.1 Encoder Evaluation

**Attribute prediction (on CelebA test set - 10k):**
```python
def evaluate_attributes(model, test_loader):
    # Per-attribute accuracy, balanced accuracy
    # Overall mean accuracy
    # Worst-5 and best-5 attributes
```

**Embedding quality:**
```python
def evaluate_embeddings(model, test_loader):
    # Compute all embeddings
    # t-SNE visualization (colored by attribute like Male/Female)
    # Retrieval metrics: Recall@K for K in [1, 5, 10]
```

### 6.2 Search Engine Evaluation

```python
def evaluate_search(search_engine, test_queries):
    # Retrieval precision@K
    # Attribute filter effectiveness
    # Query time (ms)
```

### Phase 6 Deliverables

- [ ] Attribute evaluation script
- [ ] Embedding visualization (t-SNE)
- [ ] Retrieval metrics (Recall@K)
- [ ] Search performance benchmarks

---

## Phase 7: LiteUNet Architecture & Training (Deferred)

### 7.1 Model Architecture (`src/models/unet.py`)

```
Input: (B, 4, 256, 256) → 3 RGB + 1 mask channel
          ↓
┌─────────────────────────────────────────┐
│  ENCODER                                │
│  DoubleConv(4→32) → e1 (256×256)       │
│  MaxPool → DoubleConv(32→64) → e2       │
│  MaxPool → DoubleConv(64→128) → e3      │
│  MaxPool → DoubleConv(128→256) → e4     │
│  MaxPool → DoubleConv(256→512) → bottleneck │
└─────────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────┐
│  DECODER                                │
│  Up + Concat(e4) → DoubleConv(768→256)  │
│  Up + Concat(e3) → DoubleConv(384→128)  │
│  Up + Concat(e2) → DoubleConv(192→64)   │
│  Up + Concat(e1) → DoubleConv(96→32)    │
└─────────────────────────────────────────┘
          ↓
    Conv(32→3) + Sigmoid
          ↓
Output: (B, 3, 256, 256) RGB [0, 1]
```

**DoubleConv block:**
```python
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)
```

**Channel progression (LiteUNet vs Standard):**

| Stage | Standard UNet | LiteUNet | Reduction |
|-------|---------------|----------|-----------|
| Level 1 | 64 | 32 | 2× |
| Level 2 | 128 | 64 | 2× |
| Level 3 | 256 | 128 | 2× |
| Level 4 | 512 | 256 | 2× |
| Bottleneck | 1024 | 512 | 2× |
| **Total Params** | ~31M | ~8M | **4×** |

### 7.2 Mask Generation

**Semantic masks (50% of training):**
```python
def load_semantic_mask(image_id, mask_dir, regions=['nose', 'l_eye', 'r_eye', 'mouth']):
    """Load and combine semantic masks for specified regions."""
    combined = np.zeros((256, 256), dtype=np.float32)

    for region in regions:
        mask_path = mask_dir / f"{image_id:05d}_{region}.png"
        if mask_path.exists():
            region_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            region_mask = cv2.resize(region_mask, (256, 256))
            combined = np.maximum(combined, region_mask / 255.0)

    return combined  # (256, 256) float [0, 1]
```

**Random rectangular masks (50% of training):**
```python
def generate_random_mask(image_size=256, max_area_ratio=0.4):
    """Generate random rectangular mask."""
    mask = np.zeros((image_size, image_size), dtype=np.float32)

    # Random dimensions (area < 40% of image)
    max_area = max_area_ratio * image_size * image_size
    w = np.random.randint(32, int(np.sqrt(max_area)))
    h = min(int(max_area / w), image_size - 1)

    # Random position
    x = np.random.randint(0, image_size - w)
    y = np.random.randint(0, image_size - h)

    mask[y:y+h, x:x+w] = 1.0
    return mask
```

### 7.3 Loss Functions

**L1 Loss with masked region weighting:**
```python
def weighted_l1_loss(pred, target, mask, masked_weight=5.0):
    """
    pred, target: (B, 3, H, W)
    mask: (B, 1, H, W) binary mask
    """
    # Base L1
    l1 = torch.abs(pred - target)

    # Weight masked regions higher
    weight = 1.0 + (masked_weight - 1.0) * mask
    weighted_l1 = (l1 * weight).mean()

    return weighted_l1
```

**MS-SSIM Loss:**
```python
from pytorch_msssim import ms_ssim

def msssim_loss(pred, target):
    """
    pred, target: (B, 3, H, W) in [0, 1]
    Returns: 1 - MS-SSIM (so lower is better)
    """
    msssim_val = ms_ssim(pred, target, data_range=1.0, size_average=True)
    return 1.0 - msssim_val
```

**Total loss:**
```python
loss = 1.0 * weighted_l1 + 0.2 * msssim_loss
```

### 7.4 Training Loop (`src/training/train_unet.py`)

```python
for epoch in range(num_epochs):
    model.train()

    for batch_idx, batch in enumerate(train_loader):
        image = batch['image']           # (B, 3, H, W)
        mask = batch['mask']             # (B, 1, H, W)

        # Create masked input
        masked_image = image * (1 - mask)
        input_tensor = torch.cat([masked_image, mask], dim=1)  # (B, 4, H, W)

        # Forward
        output = model(input_tensor)  # (B, 3, H, W)

        # Losses
        loss_l1 = weighted_l1_loss(output, image, mask, masked_weight=5.0)
        loss_msssim = msssim_loss(output, image)
        loss = 1.0 * loss_l1 + 0.2 * loss_msssim

        # Backward with gradient accumulation
        (loss / grad_accumulation_steps).backward()

        if (batch_idx + 1) % grad_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
```

### Phase 7 Deliverables

- [ ] `LiteUNet` class with 4-channel input
- [ ] `DoubleConv` building block
- [ ] Semantic mask loader
- [ ] Random mask generator
- [ ] Weighted L1 loss
- [ ] MS-SSIM loss (use `pytorch_msssim` package)
- [ ] Training loop with gradient accumulation
- [ ] PSNR/SSIM validation metrics
- [ ] Checkpoint saving/loading

**Run command:** `python -m src.training.train_unet --config configs/config.yaml`

---

## Phase 8: Inpainting Integration & Full Evaluation (Deferred)

### 8.1 Update Streamlit App

Replace inpainting stub with full implementation:

```python
with tab2:
    st.header("Face Inpainting")

    uploaded = st.file_uploader("Upload face image", type=['jpg', 'png'], key='inpaint')

    if uploaded:
        image = load_image(uploaded)

        region = st.selectbox("Select region to inpaint:",
                             ["Nose", "Left Eye", "Right Eye", "Mouth", "Custom Rectangle"])

        if region == "Custom Rectangle":
            # Use streamlit-drawable-canvas
            pass
        else:
            mask = get_semantic_mask(region)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(image, caption="Original")
        with col2:
            st.image(image * (1 - mask), caption="Masked")

        if st.button("Inpaint"):
            masked_input = image * (1 - mask)
            input_tensor = torch.cat([masked_input, mask], dim=0).unsqueeze(0)

            with torch.no_grad():
                output = unet(input_tensor)

            with col3:
                st.image(output[0], caption="Reconstructed")
```

### 8.2 UNet Evaluation

**On CelebA test set (10k) with semantic masks:**
```python
def evaluate_inpainting(model, test_loader):
    # PSNR (dB) - target: >25
    # SSIM - target: >0.85
    # Per-region metrics (nose, eyes, mouth)
```

### Phase 8 Deliverables

- [ ] Replace inpainting stub with full implementation
- [ ] Load UNet model in app
- [ ] Inpainting PSNR/SSIM evaluation
- [ ] Per-region inpainting quality metrics

---

## Dependencies

```txt
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
albumentations>=1.3.0
opencv-python>=4.8.0
faiss-cpu>=1.7.4
pytorch-msssim>=1.0.0
streamlit>=1.28.0
tensorboard>=2.14.0
tqdm>=4.66.0
PyYAML>=6.0
Pillow>=10.0.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
```

---

## Timeline Summary

| Phase | Status | Description | Key Outputs |
|-------|--------|-------------|-------------|
| **1** | ✅ | Data pipeline | `dataset.py`, `augmentations.py`, `config.yaml` |
| **2** | ✅ | Encoder architecture | `encoder.py` |
| **3** | ✅ | Encoder training | `train_encoder.py`, `encoder_best.pth` |
| **4** | 🔄 | Search engine | `engine.py`, `gallery_index.npz` |
| **5** | ⏳ | Streamlit app | `app.py` (search + gradcam + inpaint stub) |
| **6** | ⏳ | Evaluation (encoder/search) | Metrics, t-SNE, Recall@K |
| **7** | ⏳ | UNet training | `unet.py`, `train_unet.py`, `unet_best.pth` |
| **8** | ⏳ | Inpainting integration | Full inpainting, PSNR/SSIM |
