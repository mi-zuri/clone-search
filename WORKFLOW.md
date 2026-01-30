# Workflow Guide

## Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Verify data structure
ls data/celeba/      # 30k face images in subfolders
ls data/ffhq/        # 50k face images in subfolders
ls data/celeba_masks/ # Semantic segmentation masks
```

## Phase 1: Train Encoder

Train the face encoder with SimCLR contrastive learning + attribute prediction:

```bash
python3 -m src.training.train_encoder
```

**Output:** `checkpoints/best_encoder.pt`

**Config:** `configs/config.yaml` (batch_size, epochs, learning rates, loss weights)

**Monitor training:**
```bash
tensorboard --logdir runs/
```

## Phase 2: Build Gallery Index

Process all 80k images through encoder and build FAISS index:

```bash
python3 -m src.search.engine \
    --encoder checkpoints/best_encoder.pt \
    --output checkpoints/gallery_index.npz \
    --batch-size 64
```

**Output:** `checkpoints/gallery_index.npz` (~60MB)
- `embeddings`: (80000, 64) float32 - L2-normalized face embeddings
- `attributes`: (80000, 40) float32 - predicted attribute probabilities
- `paths`: image file paths
- `sources`: "celeba" or "ffhq"

## Phase 3: Use Search Engine

```python
import numpy as np
from src.search import FaceSearchEngine

# Load engine
engine = FaceSearchEngine("checkpoints/gallery_index.npz")
print(f"Gallery size: {len(engine)}")

# Search with random query (replace with real embedding)
query = np.random.randn(64).astype(np.float32)
query /= np.linalg.norm(query)

# Basic search
results = engine.search(query, k=5)
for r in results:
    print(f"{r['path']} - similarity: {r['similarity']:.3f}")

# Search with attribute filters
results = engine.search(
    query,
    k=5,
    attribute_filters={"Smiling": True, "Eyeglasses": False}
)
```

**Available attributes:** `engine.ATTRIBUTE_NAMES` (40 CelebA attributes)

## Run Tests

```bash
# All tests
python3 -m pytest tests/ -v

# Specific modules
python3 -m pytest tests/test_encoder.py -v
python3 -m pytest tests/test_search.py -v
python3 -m pytest tests/test_training.py -v
```

## Quick Verification

```bash
# Check encoder output shapes
python3 -c "
import torch
from src.models.encoder import FaceEncoder

model = FaceEncoder(embedding_dim=64, projection_dim=64, num_attributes=40)
x = torch.randn(2, 3, 256, 256)
emb, attr = model(x)
print(f'Embedding: {emb.shape}')  # [2, 64]
print(f'Attributes: {attr.shape}')  # [2, 40]
"

# Check gallery index
python3 -c "
import numpy as np
data = np.load('checkpoints/gallery_index.npz')
print(f'Embeddings: {data[\"embeddings\"].shape}')
print(f'Attributes: {data[\"attributes\"].shape}')
print(f'Total images: {len(data[\"paths\"])}')
"
```

## Data Splits

Training used fixed index-based splits:

| Dataset | Split | Indices | Count |
|---------|-------|---------|-------|
| CelebA | train | 0-14999 | 15,000 |
| CelebA | val | 15000-19999 | 5,000 |
| CelebA | test | 20000-29999 | 10,000 |
| FFHQ | train | 0-4999 | 5,000 |
| FFHQ | test | 5000+ | ~45,000 |

**View split info:**
```bash
python3 -m src.search.splits
```

**Test with unseen images:**
```python
import torch
import numpy as np
from src.search import FaceSearchEngine
from src.search.splits import get_test_datasets
from src.models.encoder import FaceEncoder

# Load model and search engine
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
checkpoint = torch.load("checkpoints/best_encoder.pt", map_location=device)
model = FaceEncoder(embedding_dim=64, projection_dim=64, num_attributes=40)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device).eval()

engine = FaceSearchEngine("checkpoints/gallery_index.npz")

# Get test datasets (images NOT used in training)
celeba_test, ffhq_test = get_test_datasets()

# Encode a test image and search
sample = celeba_test[0]
with torch.no_grad():
    img = sample["image"].unsqueeze(0).to(device)
    embedding, _ = model(img)
    query = embedding.cpu().numpy()

results = engine.search(query, k=5)
print(f"Query: {sample['path']}")
for r in results:
    print(f"  {r['similarity']:.3f} - {r['path']}")
```

## Troubleshooting

**OpenMP conflict (macOS):**
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

**MPS memory issues:**
Reduce batch size in `configs/config.yaml`

**Missing files in FFHQ:**
Dataset now scans actual files, handles non-sequential numbering automatically.
