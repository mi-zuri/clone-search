# Face Search

Computer Vision Project — multi-task face encoder (SimCLR contrastive learning + 40-attribute prediction) with FAISS-backed similarity search over an 80k-image gallery (CelebA-HQ + FFHQ).

---

![search preview](docs/images/preview.png)

---

## Features

- **Face Search**: Find similar faces using learned 64-D embeddings + FAISS cosine search
- **Attribute Prediction**: 40 CelebA binary attributes, jointly trained with the embedding head
- **Attribute Filtering**: Filter retrieval results by predicted attributes (e.g. `Smiling=True, Eyeglasses=False`)
- **Grad-CAM**: Visual explanation of encoder attention in the Streamlit app
- **Evaluation suite**: Retrieval metrics, attribute accuracy, embedding visualizations, and search benchmarks

Attribute prediction:

![man with a hat](docs/images/hat.png)

Matching attributes:

![bald woman](docs/images/bald.png)

> **Note**: Face inpainting (e.g. masked-region reconstruction with a U-Net) is **not implemented** — the Streamlit app contains a placeholder tab only. The architecture and config leave room to add it later as a second training stage on top of the existing encoder/data pipeline.

## Project Structure

```
CV_clone_search/
├── src/
│   ├── data/
│   │   ├── dataset.py            # CelebA-HQ & FFHQ loaders
│   │   └── augmentations.py      # albumentations pipeline (SimCLR views + val)
│   ├── models/
│   │   └── encoder.py            # Face encoder (embedding + attribute heads)
│   ├── training/
│   │   └── train_encoder.py      # SimCLR + attribute multi-task training
│   ├── search/
│   │   ├── engine.py             # FAISS search engine + gallery indexer
│   │   └── splits.py             # Fixed train/val/test index splits
│   ├── evaluation/
│   │   ├── evaluate_retrieval.py # Retrieval metrics
│   │   ├── evaluate_attributes.py# Per-attribute accuracy / F1
│   │   ├── benchmark_search.py   # FAISS latency benchmarks
│   │   └── visualize_embeddings.py
│   ├── visualization/
│   │   └── generate_report_figures.py
│   └── app.py                    # Streamlit GUI
├── configs/
│   └── config.yaml               # Hyperparameters
├── data/                         # Datasets (gitignored)
├── checkpoints/                  # best_encoder.pt, gallery_index.npz
├── runs/                         # TensorBoard logs
├── results/                      # Eval outputs (JSON / CSV / figures)
└── docs/
    ├── REPORT.md / REPORT.pdf    # Project report
    └── ...                       # Plans, system overview, grading notes
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

## Evaluation

After training and indexing, reproduce the numbers under `results/`:

```bash
python3 -m src.evaluation.evaluate_retrieval
python3 -m src.evaluation.evaluate_attributes
python3 -m src.evaluation.benchmark_search
python3 -m src.evaluation.visualize_embeddings
```

## Configuration

Edit `configs/config.yaml`. Key sections:

```yaml
encoder:
  batch_size: 32
  gradient_accumulation_steps: 8   # effective batch = 256 for SimCLR
  epochs: 6
  lr: 0.001
  embedding_dim: 64
  num_attributes: 40
  simclr_temperature: 0.25
  simclr_loss_weight: 0.5
  attribute_loss_weight: 1.0

data:
  image_size: 256
  celeba_train: 15000
  celeba_val: 5000
  celeba_test: 10000
  ffhq_train_subset: 5000
  celeba_batch_ratio: 0.75         # 75% CelebA + 25% FFHQ per batch

search:
  top_k: 5
  use_faiss: true
  gallery_size: 80000              # 30k CelebA-HQ + 50k FFHQ
```

## Requirements

- Python 3.11+
- PyTorch 2.0+
- CUDA (recommended) or MPS (Apple Silicon)
