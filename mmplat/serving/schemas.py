from typing import Optional, Dict, Any
from pydantic import BaseModel

class JSONPredictRequest(BaseModel):
    text: Optional[str] = None
    tabular: Optional[Dict[str, Any]] = None
    image_b64: Optional[str] = None  # base64-encoded PNG/JPEG

class PredictResponse(BaseModel):
    prediction: Any
    confidence: float
    raw: Dict[str, Any]
