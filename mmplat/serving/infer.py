import io, base64, torch, numpy as np
from typing import Dict, Any, Optional
from PIL import Image
import yaml, os

from mmplat.preprocessing.image import ImagePreprocessor
from mmplat.preprocessing.tabular import TabularPreprocessor
from mmplat.preprocessing.text import TextPreprocessor
from mmplat.models.image_encoder import ImageEncoder
from mmplat.models.tabular_encoder import TabularEncoder
from mmplat.models.text_encoder import TextEncoderHF, TextEncoderBOW
from mmplat.models.fusion import MultiModalModel, PredictionHead
from scipy import sparse 

def load_checkpoint(ckpt_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)

    # Load preprocessors
    img_conf = cfg.get("preprocessing",{}).get("image",{})
    tab_conf = cfg.get("preprocessing",{}).get("tabular",{})
    txt_conf = cfg.get("preprocessing",{}).get("text",{})
    image_pre = ImagePreprocessor(size=img_conf.get("size",224),
                                  mean=img_conf.get("normalize_mean"),
                                  std=img_conf.get("normalize_std"),
                                  augment=False)
    tab_pre = TabularPreprocessor(num_cols=cfg["data"]["num_cols"], cat_cols=cfg["data"]["cat_cols"],
                                  save_path=tab_conf.get("save_path"))
    tab_pre.load()
    text_pre = TextPreprocessor(use_transformer=txt_conf.get("use_transformer", True),
                                model_name=txt_conf.get("model_name","distilbert-base-uncased"),
                                max_length=txt_conf.get("max_length",128))
    return model, cfg, image_pre, tab_pre, text_pre, device

def build_model(cfg):
    # tab input dim must match preprocessing. Reconstruct from saved preprocessor on-the-fly
    from joblib import load
    tab_pre_path = cfg["preprocessing"]["tabular"]["save_path"]
    ct = load(tab_pre_path)
    # quick way: transform empty row to get output dim
    import pandas as pd, numpy as np
    dummy = ct.transform(pd.DataFrame([{**{c: None for c in cfg["data"]["num_cols"] + cfg["data"]["cat_cols"]}}]))
    try:
        dummy = dummy.toarray()
    except AttributeError:
        pass
    dummy = np.asarray(dummy).squeeze(0)
    tab_in_dim = int(dummy.shape[0])

    # Build encoders
    img_enc = None
    if cfg["model"].get("image_encoder"):
        ic = cfg["model"]["image_encoder"]
        img_enc = ImageEncoder(ic.get("name","resnet18"), ic.get("pretrained", True), ic.get("out_dim",128))
    tab_enc = None
    if cfg["model"].get("tabular_encoder"):
        tc = cfg["model"]["tabular_encoder"]
        tab_enc = TabularEncoder(tab_in_dim, tc.get("hidden_dims",[64,64]), tc.get("out_dim",64), tc.get("dropout",0.1))
    txt_enc = None
    if cfg["model"].get("text_encoder") is not False:
        out_dim = cfg["model"]["text_encoder"].get("out_dim", 128)
        if cfg["preprocessing"]["text"].get("use_transformer",True):
            txt_enc = TextEncoderHF(cfg["preprocessing"]["text"].get("model_name","distilbert-base-uncased"), out_dim)
        else:
            txt_enc = TextEncoderBOW(1024, out_dim)

    # Fusion + head
    in_dims = []
    if img_enc is not None: in_dims.append(cfg["model"]["image_encoder"].get("out_dim", 128))
    if tab_enc is not None: in_dims.append(cfg["model"]["tabular_encoder"].get("out_dim", 64))
    if txt_enc is not None: in_dims.append(cfg["model"]["text_encoder"].get("out_dim", 128))
    from mmplat.models.fusion import PredictionHead, MultiModalModel
    head = PredictionHead(
        sum(in_dims),
        cfg["model"]["head"].get("hidden_dims",[128]),
        cfg.get("num_classes",1) if cfg["task_type"]=="classification" else 1,
        cfg["task_type"],
        cfg["model"]["head"].get("dropout",0.1)
    )

    fusion_cfg = cfg["model"].get("fusion", "concat")
    fusion_type = fusion_cfg.get("type", "concat") if isinstance(fusion_cfg, dict) else fusion_cfg

    model = MultiModalModel(img_enc, tab_enc, txt_enc, fusion_type, head, cfg["task_type"])
    return model

def preprocess_inputs(image_pre, tab_pre, text_pre, cfg, image=None, text: Optional[str]=None, tabular: Optional[Dict[str,Any]]=None, device="cpu") -> Dict[str,Any]:
    import torch, numpy as np
    out = {}

    # Image: always provide a tensor if image encoder is configured
    if cfg.get("model", {}).get("image_encoder"):
        # ImagePreprocessor(image=None) -> zero image tensor by design
        img_t = image_pre(image)
        out["image_tensor"] = img_t.unsqueeze(0).to(device)

    # Tabular: always provide a tensor if tabular encoder is configured
    if cfg.get("model", {}).get("tabular_encoder"):
        arr = tab_pre.transform(tabular)  # tabular can be None; pipeline imputes
        out["tabular_tensor"] = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(device)

    # Text: always provide tokens/BOW if text encoder is configured
    if cfg.get("model", {}).get("text_encoder", True) is not False:
        if text_pre.use_transformer:
            toks = text_pre(text or "")
            out["input_ids"] = toks["input_ids"].unsqueeze(0).to(device)
            out["attention_mask"] = toks["attention_mask"].unsqueeze(0).to(device)
        else:
            out["bow"] = torch.tensor(text_pre(text or "")["bow"], dtype=torch.float32).unsqueeze(0).to(device)

    return out
    
def decode_b64_image(image_b64: str) -> Optional[str]:
    try:
        binary = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(binary)).convert("RGB")
        # Save to a temporary path
        tmp_path = "/tmp/upload.png"
        img.save(tmp_path)
        return tmp_path
    except Exception:
        return None
