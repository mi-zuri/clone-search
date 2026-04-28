# Face Search & Inpainting

Computer Vision Project - Multi-task learning for face similarity search and semantic inpainting.

---

![search preview](docs/images/preview.png)

---

## Features

- **Face Search**: Find similar faces using learned embeddings + FAISS
- **Face Inpainting**: Remove and reconstruct facial features (eyes, nose, mouth)
- **Attribute Prediction**: 40 CelebA binary attributes
- **Grad-CAM**: Visual explanation of encoder attention

Attribute prediction:

![man with a hat](docs/images/hat.png)

Matching attributes:

![bald woman](docs/images/bald.png)

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

## Running the Streamlit app

The Streamlit UI entrypoint is `src/app.py`.

### 1) Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)

pip install -r requirements.txt
```

### 2) Prepare required artifacts

The app expects these files to exist:

- `checkpoints/best_encoder.pt`
- `checkpoints/gallery_index.npz`

You can produce them by following the steps in [`WORKFLOW.md`](WORKFLOW.md):
1) train the encoder (`python3 -m src.training.train_encoder`)
2) build the gallery index (`python3 -m src.search.engine ...`)

If they are missing, the app will show a message like:
> Make sure checkpoints/best_encoder.pt and checkpoints/gallery_index.npz exist.

### 3) Start the app

Run Streamlit from the repository root:

```bash
streamlit run src/app.py
```

Then open the local URL printed by Streamlit (typically `http://localhost:8501`).

### Notes / troubleshooting

- On macOS, an OpenMP conflict between PyTorch and FAISS is worked around in the app by setting `KMP_DUPLICATE_LIB_OK=TRUE`.
- The app will auto-select the best available device in this order: **MPS > CUDA > CPU**.

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
