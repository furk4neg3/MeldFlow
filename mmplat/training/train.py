import os, yaml, math
from typing import Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from mmplat.data_loaders.multimodal_dataset import MultiModalDataset
from mmplat.preprocessing.image import ImagePreprocessor
from mmplat.preprocessing.tabular import TabularPreprocessor
from mmplat.preprocessing.text import TextPreprocessor
from mmplat.models.image_encoder import ImageEncoder
from mmplat.models.tabular_encoder import TabularEncoder
from mmplat.models.text_encoder import TextEncoderHF, TextEncoderBOW
from mmplat.models.fusion import MultiModalModel, PredictionHead
from mmplat.training.utils import set_seed, classification_metrics, regression_metrics, save_checkpoint, write_log_row

def _as_float(x, default):
    try:
        return float(x)
    except Exception:
        return float(default)

def _as_int(x, default):
    try:
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return int(default)

def _sanitize_cfg_for_ckpt(cfg: dict) -> dict:
    """Drop runtime objects (like preprocessors) from cfg before saving."""
    allowed = ["seed","task_type","num_classes","target_column","split","data","preprocessing","model","training"]
    return {k: cfg[k] for k in allowed if k in cfg}

def collate_fn(batch: List[Dict[str, Any]], device: str, cfg: Dict):
    import torch
    B = len(batch)
    out = {}

    # Image
    if cfg.get("model", {}).get("image_encoder") is not None:
        imgs = torch.stack([cfg["_image_preproc"](b.get("image_path")) for b in batch], dim=0)
        out["image_tensor"] = imgs.to(device)

    # Tabular
    if cfg.get("model", {}).get("tabular_encoder") is not None:
        tab = [b.get("tabular") for b in batch]
        arrs = [cfg["_tab_preproc"].transform(t) for t in tab]
        tab_t = torch.tensor(np.stack(arrs, axis=0), dtype=torch.float32)
        out["tabular_tensor"] = tab_t.to(device)

    # Text
    if cfg.get("model", {}).get("text_encoder", True) is not False:
        text_prep = cfg.get("preprocessing", {}).get("text", {})
        use_transformer = text_prep.get("use_transformer", True)
        if use_transformer:
            toks = [cfg["_text_preproc"](b.get("text")) for b in batch]
            input_ids = torch.stack([t["input_ids"] for t in toks], dim=0)
            attn = torch.stack([t["attention_mask"] for t in toks], dim=0)
            out["input_ids"] = input_ids.to(device)
            out["attention_mask"] = attn.to(device)
        else:
            bows = [cfg["_text_preproc"](b.get("text"))["bow"] for b in batch]
            out["bow"] = torch.tensor(np.stack(bows, axis=0), dtype=torch.float32).to(device)

    # Targets
    ys = [b.get("y") for b in batch]
    if cfg["task_type"] == "classification":
        y_t = torch.tensor([int(y) for y in ys], dtype=torch.long).to(device)
    else:
        y_t = torch.tensor([float(y) for y in ys], dtype=torch.float32).to(device)
    out["y"] = y_t
    return out

def build_model(cfg: Dict, tab_in_dim: int) -> MultiModalModel:
    img_enc = None
    if cfg["model"].get("image_encoder"):
        img_conf = cfg["model"]["image_encoder"]
        img_enc = ImageEncoder(
            name=img_conf.get("name","resnet18"),
            pretrained=img_conf.get("pretrained", True),
            out_dim=img_conf.get("out_dim", 128)
        )

    tab_enc = None
    if cfg["model"].get("tabular_encoder"):
        tconf = cfg["model"]["tabular_encoder"]
        tab_enc = TabularEncoder(
            in_dim=tab_in_dim,
            hidden_dims=tconf.get("hidden_dims", [64,64]),
            out_dim=tconf.get("out_dim", 64),
            dropout=tconf.get("dropout", 0.1)
        )

    txt_enc = None
    if cfg["model"].get("text_encoder") is not False:
        out_dim = cfg["model"]["text_encoder"].get("out_dim", 128)
        text_prep = cfg.get("preprocessing", {}).get("text", {})
        use_transformer = text_prep.get("use_transformer", True)
        model_name = text_prep.get("model_name", "distilbert-base-uncased")
        if use_transformer:
            txt_enc = TextEncoderHF(model_name=model_name, out_dim=out_dim)
        else:
            txt_enc = TextEncoderBOW(in_dim=1024, out_dim=out_dim)

    # Fusion input dim
    in_dims = []
    if img_enc is not None: in_dims.append(cfg["model"]["image_encoder"].get("out_dim", 128))
    if tab_enc is not None: in_dims.append(cfg["model"]["tabular_encoder"].get("out_dim", 64))
    if txt_enc is not None: in_dims.append(cfg["model"]["text_encoder"].get("out_dim", 128))
    fusion_in = sum(in_dims)

    head = PredictionHead(
        in_dim=fusion_in,
        hidden_dims=cfg["model"]["head"].get("hidden_dims", [128]),
        out_dim=cfg.get("num_classes", 1) if cfg["task_type"]=="classification" else 1,
        task_type=cfg["task_type"],
        dropout=cfg["model"]["head"].get("dropout", 0.1)
    )

    fusion_cfg = cfg["model"].get("fusion", "concat")
    fusion_type = fusion_cfg.get("type", "concat") if isinstance(fusion_cfg, dict) else fusion_cfg

    model = MultiModalModel(
        image_encoder=img_enc,
        tabular_encoder=tab_enc,
        text_encoder=txt_enc,
        fusion=fusion_type,
        head=head,
        task_type=cfg["task_type"]
    )
    return model

def main(config_path: str):
    import pandas as pd
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 1337))

    # Create datasets (use split column if present)
    split_cfg = cfg.get("split", {})
    split_col = split_cfg.get("split_column")

    common_ds_kwargs = dict(
        csv_path = cfg["data"]["csv_path"],
        image_column = cfg["data"]["image_column"],
        text_column = cfg["data"]["text_column"],
        num_cols = cfg["data"]["num_cols"],
        cat_cols = cfg["data"]["cat_cols"],
        target_column = cfg["target_column"],
        image_root = cfg["data"].get("image_root"),
        drop_all_missing = cfg["data"].get("drop_all_missing", True),
        split_column = split_col,
    )

    if split_col:
        train_ds = MultiModalDataset(split="train", **common_ds_kwargs)
        val_ds   = MultiModalDataset(split="val", **common_ds_kwargs)
        test_ds  = MultiModalDataset(split="test", **common_ds_kwargs)
    else:
        df = pd.read_csv(cfg["data"]["csv_path"]).sample(frac=1.0, random_state=cfg.get("seed",1337)).reset_index(drop=True)
        n = len(df)
        tr = int(n*cfg["split"].get("train_ratio",0.7))
        vr = int(n*cfg["split"].get("val_ratio",0.15))
        df.loc[:tr-1, "split"] = "train"
        df.loc[tr:tr+vr-1, "split"] = "val"
        df.loc[tr+vr:, "split"] = "test"
        df.to_csv(cfg["data"]["csv_path"], index=False)
        train_ds = MultiModalDataset(split="train", **common_ds_kwargs)
        val_ds   = MultiModalDataset(split="val", **common_ds_kwargs)
        test_ds  = MultiModalDataset(split="test", **common_ds_kwargs)

    # Preprocessors
    img_conf = cfg.get("preprocessing",{}).get("image",{})
    tab_conf = cfg.get("preprocessing",{}).get("tabular",{})
    txt_conf = cfg.get("preprocessing",{}).get("text",{})

    _image_preproc = ImagePreprocessor(
        size=img_conf.get("size",224),
        mean=img_conf.get("normalize_mean"),
        std=img_conf.get("normalize_std"),
        augment=img_conf.get("augment",True)
    )

    _tab_preproc = TabularPreprocessor(
        num_cols=cfg["data"]["num_cols"],
        cat_cols=cfg["data"]["cat_cols"],
        save_path=tab_conf.get("save_path")
    )
    _tab_preproc.fit([train_ds[i]["tabular"] for i in range(len(train_ds))])
    _tab_preproc.save()
    tab_in_dim = _tab_preproc.output_dim()

    _text_preproc = TextPreprocessor(
        use_transformer=txt_conf.get("use_transformer", True),
        model_name=txt_conf.get("model_name","distilbert-base-uncased"),
        max_length=txt_conf.get("max_length",128)
    )

    # Attach preprocessors for collate_fn
    cfg["_image_preproc"] = _image_preproc
    cfg["_tab_preproc"] = _tab_preproc
    cfg["_text_preproc"] = _text_preproc

    # Build model
    model = build_model(cfg, tab_in_dim=tab_in_dim).to(device)

    # Cast/normalize all training hparams
    bs = _as_int(cfg.get("training", {}).get("batch_size", 32), 32)
    num_workers = _as_int(cfg.get("training", {}).get("num_workers", 2), 2)
    num_epochs = _as_int(cfg.get("training", {}).get("num_epochs", 5), 5)
    lr = _as_float(cfg.get("training", {}).get("lr", 2e-4), 2e-4)
    weight_decay = _as_float(cfg.get("training", {}).get("weight_decay", 1e-4), 1e-4)
    patience = _as_int(cfg.get("training", {}).get("patience", 3), 3)

    # Dataloaders
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=num_workers,
                              collate_fn=lambda b: collate_fn(b, device, cfg))
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                              num_workers=num_workers,
                              collate_fn=lambda b: collate_fn(b, device, cfg))
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False,
                              num_workers=num_workers,
                              collate_fn=lambda b: collate_fn(b, device, cfg))

    # Criterion & optimizer
    task = cfg.get("task_type","classification")
    if task == "classification":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = None
    bad_epochs = 0

    os.makedirs(cfg["training"].get("save_dir","artifacts"), exist_ok=True)
    log_csv = cfg["training"].get("log_csv","artifacts/train_log.csv")

    for epoch in range(1, num_epochs+1):
        model.train()
        ep_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
            y = batch["y"]
            logits = model(batch)
            if task == "classification":
                loss = criterion(logits, y)
            else:
                loss = criterion(logits.squeeze(), y.float())
            optim.zero_grad()
            loss.backward()
            optim.step()
            ep_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        metric_val = 0.0
        n_samples = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
                y = batch["y"]
                logits = model(batch)
                if task == "classification":
                    loss = criterion(logits, y)
                    metrics = classification_metrics(logits, y)
                    metric_val += metrics["accuracy"] * y.size(0)
                else:
                    loss = criterion(logits.squeeze(), y.float())
                    preds = logits.detach()
                    metrics = regression_metrics(preds, y)
                    metric_val += (-metrics["mse"]) * y.size(0)  # higher is better
                val_loss += loss.item()
                n_samples += y.size(0)

        score = metric_val / max(1, n_samples)
        write_log_row(log_csv, {"epoch": epoch, "train_loss": ep_loss, "val_loss": val_loss, "score": score})

        improve = (best_val is None) or (score > best_val)
        if improve:
            best_val = score
            clean_cfg = _sanitize_cfg_for_ckpt(cfg)
            save_checkpoint(cfg["training"].get("best_ckpt","artifacts/best_model.pt"), {
                "model_state": model.state_dict(),
                "cfg": clean_cfg,
            })
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping.")
                break

    # Final test
    ckpt = torch.load(cfg["training"].get("best_ckpt","artifacts/best_model.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    n_samples = 0
    metric_val = 0.0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test"):
            y = batch["y"]
            logits = model(batch)
            if task == "classification":
                metrics = classification_metrics(logits, y)
                metric_val += metrics["accuracy"] * y.size(0)
            else:
                preds = logits.detach()
                metrics = regression_metrics(preds, y)
                metric_val += (-metrics["mse"]) * y.size(0)
            n_samples += y.size(0)
    score = metric_val / max(1, n_samples)
    print(f"Test score: {score:.4f}")
    return score

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config/default.yaml")
    args = ap.parse_args()
    main(args.config)
