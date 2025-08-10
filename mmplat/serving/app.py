from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any
import os, io, base64, json, torch, numpy as np

from mmplat.serving.schemas import JSONPredictRequest
from mmplat.serving.infer import load_checkpoint, preprocess_inputs, decode_b64_image

CKPT_PATH = os.getenv("MM_CKPT", "artifacts/best_model.pt")
model, cfg, image_pre, tab_pre, text_pre, device = load_checkpoint(CKPT_PATH)

app = FastAPI(title="Multi-Modal Analytics Platform", version="0.1.0")


@app.post("/predict")  # accepts JSON with optional image_b64
async def predict_json(req: JSONPredictRequest):
    image_path = None
    if req.image_b64:
        image_path = decode_b64_image(req.image_b64)
    batch = preprocess_inputs(image_pre, tab_pre, text_pre, cfg, image=image_path, text=req.text, tabular=req.tabular, device=device)
    with torch.no_grad():
        logits = model(batch)
        if cfg["task_type"] == "classification":
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            pred = int(probs.argmax())
            conf = float(probs.max())
        else:
            pred = float(logits.squeeze().cpu().numpy())
            conf = 1.0
    return {"prediction": pred, "confidence": conf, "raw": {"logits": logits.squeeze().tolist()}}


@app.post("/predict-multipart")  # multipart form-data: image file + fields
async def predict_multipart(
    image: Optional[UploadFile] = File(default=None),
    text: Optional[str] = Form(default=None),
    tabular_json: Optional[str] = Form(default=None),
):
    image_path = None
    if image is not None:
        content = await image.read()
        import PIL.Image as PImage, io as iio, uuid, os
        tmp_path = f"/tmp/{uuid.uuid4().hex}.png"
        PImage.open(iio.BytesIO(content)).convert("RGB").save(tmp_path)
        image_path = tmp_path
    tabular = json.loads(tabular_json) if tabular_json else None
    batch = preprocess_inputs(image_pre, tab_pre, text_pre, cfg, image=image_path, text=text, tabular=tabular, device=device)
    with torch.no_grad():
        logits = model(batch)
        if cfg["task_type"] == "classification":
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            pred = int(probs.argmax())
            conf = float(probs.max())
        else:
            pred = float(logits.squeeze().cpu().numpy())
            conf = 1.0
    return {"prediction": pred, "confidence": conf, "raw": {"logits": logits.squeeze().tolist()}}
