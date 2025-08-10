import os, csv, math, random
import numpy as np
import torch
from typing import Dict

def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def classification_metrics(logits, y_true):
    # logits: [B, C], y_true: [B]
    with torch.no_grad():
        preds = torch.argmax(logits, dim=-1)
        correct = (preds == y_true).sum().item()
        acc = correct / max(1, y_true.size(0))
    return {"accuracy": acc}

def regression_metrics(preds, y_true):
    with torch.no_grad():
        mse = torch.mean((preds.squeeze() - y_true.float())**2).item()
        mae = torch.mean(torch.abs(preds.squeeze() - y_true.float())).item()
    return {"mse": mse, "mae": mae}

def save_checkpoint(path: str, state: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

def write_log_row(csv_path: str, row: Dict):
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)
