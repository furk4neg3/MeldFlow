# MeldFlow

A production‑grade, extensible toolkit for training and serving multi‑modal models across **images + tabular + text**.

> Train on mixed modalities. Fuse embeddings. Serve with FastAPI.

---

## *Why* MeldFlow?

MeldFlow (Multi‑Modal Analytics Platform) gives you a clean, batteries‑included path from ****data → training → checkpoints → APIs****. It supports missing modalities, configurable encoders, and a simple FastAPI service for low‑latency inference.

## Features

* **Three modalities** out of the box: image, tabular, text (transformer or BOW).
* **Pluggable encoders:** ResNet for vision, MLP for tabular, HF Transformers for text.
* **Configurable fusion** (concat by default) and a lightweight prediction head for classification or regression.
* **Training loop** with splits, metrics, early stopping, checkpoints, and CSV logs.
* **FastAPI inference service** accepting JSON (base64 image) or multipart (file upload).
* **Synthetic dataset generator** for sanity checks and CI.
* **Dockerfile, tests, and YAML config** to keep things reproducible.

---

## Quick test (Single Command)

Run the entire smoke suite with a single command:

```bash
make test
```

What this does:

* Starts Docker and installs deps.
* Generates a tiny synthetic dataset.
* Runs unit tests and a fast train→infer smoke test on the multi‑modal pipeline.

If `make` isn’t available on your system, run:

```bash
python -m pytest -q
```

---

## Quickstart (Docker‑first)

### 1) Build the image

```bash
docker build -t mm-analytics:latest .
```

### 2) Generate example data

```bash
docker run --rm -v "$PWD":/app mm-analytics \
  python scripts/generate_synth_dataset.py --out_dir data/synth --num_samples 600 --image_size 64
```

### 3) Train

```bash
docker run --rm --shm-size=1g \
  -v "$PWD":/app -v hf_cache:/root/.cache/huggingface \
  mm-analytics \
  python scripts/train.py --config config/default.yaml
```

### 4) Serve the API

```bash
docker run --rm -p 8000:8000 -v "$PWD":/app \
  -e MM_CKPT=artifacts/best_model.pt \
  mm-analytics
```

### 5) Test the API

```bash
# multipart without image
curl -X POST http://127.0.0.1:8000/predict-multipart \
  -F text="red square low value" \
  -F 'tabular_json={"num_a":0.1,"num_b":2.0,"cat_x":"A"}'

# multipart with image
curl -X POST http://127.0.0.1:8000/predict-multipart \
  -F image=@data/synth/images/sample_0.png
```

---

## Local quickstart (no Docker)

### **1) Environment (Python 3.11)**

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### **2) Create example data**

```bash
python scripts/generate_synth_dataset.py --out_dir data/synth --num_samples 600 --image_size 64
```

This writes `data/synth/synth.csv` and `data/synth/images/` with aligned image/text/tabular rows and a `split` column.

### 3) Train

```bash
python scripts/train.py --config config/default.yaml
```

Artifacts (best checkpoint, preprocessors, and logs) land in `artifacts/` by default.

### 4) Serve an API

Start the server (uses `MM_CKPT` if set, otherwise `artifacts/best_model.pt`):

```bash
uvicorn mmplat.serving.app:app --reload --host 0.0.0.0 --port 8000
# optional
export MM_CKPT=artifacts/best_model.pt
```

#### JSON endpoint: `/predict`

**Request body** (any field optional):

```json
{
  "text": "green circle medium value",
  "tabular": {"num_a": 0.4, "num_b": 1.2, "cat_x": "A"},
  "image_b64": "<base64-encoded PNG/JPEG>"
}
```

**Response**

```json
{"prediction": 1, "confidence": 0.93, "raw": {"logits": [..]}}
```

#### Multipart endpoint: `/predict-multipart`

Form fields:

* `image`: file (optional)
* `text`: string (optional)
* `tabular_json`: JSON string (optional)

Example client:

```bash
python scripts/infer_example.py --host http://127.0.0.1:8000 \
  --image data/synth/images/sample.png \
  --text "blue triangle high value" \
  --tabular_json '{"num_a": 0.8, "num_b": 0.3, "cat_x": "C"}'
```

### 5) Docker

````bash
docker build -t mm-analytics:latest .
docker run -p 8000:8000 -v "$PWD":/app -e MM_CKPT=artifacts/best_model.pt mm-analytics
docker build -t meldflow .
docker run -p 8000:8000 -e MM_CKPT=artifacts/best_model.pt meldflow
````

---

## Configuration

All knobs live in `config/default.yaml`. A representative subset:

```yaml
seed: 1337
task_type: classification   # or regression
num_classes: 3              # classification only

data:
  csv_path: data/synth/synth.csv
  image_root: data/synth/images
  image_column: image_path
  text_column: text
  num_cols: [num_a, num_b]
  cat_cols: [cat_x]
  target_column: label
  split:
    split_column: split       # or use ratios below
    train_ratio: 0.7
    val_ratio: 0.15
    test_ratio: 0.15

preprocessing:
  image:
    size: 224
    normalize_mean: [0.485, 0.456, 0.406]
    normalize_std:  [0.229, 0.224, 0.225]
  tabular:
    save_path: artifacts/tabular_preproc.joblib  # persisted ColumnTransformer
  text:
    use_transformer: true
    model_name: distilbert-base-uncased
    max_length: 128

model:
  image_encoder:   { name: resnet18, pretrained: true,  out_dim: 128 }
  tabular_encoder: { hidden_dims: [64,64],               out_dim: 64, dropout: 0.1 }
  text_encoder:    { use_transformer: true,              out_dim: 128 }
  fusion: concat
  head:            { hidden_dims: [128], dropout: 0.1 }

training:
  batch_size: 32
  num_epochs: 5
  lr: 2e-4
  weight_decay: 1e-4
  patience: 3
  save_dir: artifacts
  best_ckpt: artifacts/best_model.pt
  log_csv: artifacts/train_log.csv
```

> 🔎 Tip: set `text.use_transformer: false` to switch to a lightweight BOW encoder (offline, fast prototyping).

---

## Your data: expected CSV schema

MeldFlow reads a single CSV with at least these columns:

| column         | type           | notes                                     |
| -------------- | -------------- | ----------------------------------------- |
| `image_path`   | string         | relative to `data.image_root` or absolute |
| `text`         | string         | free text                                 |
| `num_a`, `...` | float          | numeric columns (configure in YAML)       |
| `cat_x`, `...` | string         | categorical columns (configure in YAML)   |
| `label`        | int/float      | target (class id or regression value)     |
| `split`        | train/val/test | optional; otherwise ratios are used       |

Rows with **all modalities missing** are automatically dropped.

---

## Extending MeldFlow

* **Encoders:** swap ResNet depth, add ViT, or replace tabular MLP.
* **Text:** change HF model via `text.model_name` or use BOW fallback.
* **Fusion:** add strategies in `mmplat/models/fusion.py` (e.g., gated/attention).
* **Heads:** adapt the MLP head or implement multi‑task heads.

---

## Tests

Prefer the Makefile target:

```bash
make test
```

Or run PyTest directly:

```bash
pytest -q
```

---

## Project layout

```
mmplat/
  data_loaders/
  preprocessing/
  models/
  training/
  serving/
config/
scripts/
tests/
artifacts/
data/
```

---

## Requirements

* Python 3.11
* See `requirements.txt` for libraries: PyTorch, torchvision, transformers, scikit‑learn, pandas, numpy, Pillow, FastAPI, Uvicorn, etc.

---

## Performance & tips

* Enable CUDA if available; checkpoints load on CPU/GPU automatically.
* Synthetic data is **predictable** to help with quick sanity checks.
* Missing modalities are handled via masking/zero‑fill in fused embeddings.

---

## Versioning & naming

* Package version: see `mmplat/version.py` (e.g., `0.1.0`).
* Current project name: **MeldFlow**. If you prefer to keep the `mmplat` package name for imports, that’s fine—the readme branding doesn’t break code.

---

## License

Licensed under MIT License.

---

## Acknowledgments

Built on PyTorch, torchvision, scikit‑learn, Hugging Face Transformers, and FastAPI.

---

## Mermaid architecture

```mermaid
flowchart LR
    A[Raw Inputs] -->|image| I[Image Preproc]
    A -->|tabular| T[Tabular Preproc]
    A -->|text| X[Text Preproc]

    I --> IE[Image Encoder (ResNet*)]
    T --> TE[Tabular Encoder (MLP)]
    X --> XE[Text Encoder (Transformer/BOW)]

    IE --> F[Fusion (Concat*)]
    TE --> F
    XE --> F

    F --> H[Head (Cls/Reg)]
    H --> O[Prediction + Confidence]

    classDef dim fill:#f6f8fa,stroke:#d0d7de,stroke-width:1px;
    class A,I,T,X,IE,TE,XE,F,H,O dim;
```
