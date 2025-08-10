from typing import Optional, Dict, Any
import numpy as np

class TextPreprocessor:
    """Wraps HF tokenizer if use_transformer is True; else BOW fallback (very light).
    """
    def __init__(self, use_transformer: bool = True, model_name: str = "distilbert-base-uncased", max_length: int = 128):
        self.use_transformer = use_transformer
        self.model_name = model_name
        self.max_length = max_length
        if self.use_transformer:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            self.vocab = {}

    def __call__(self, text: Optional[str]) -> Dict[str, Any]:
        if self.use_transformer:
            if not text:
                text = ""
            toks = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            return {"input_ids": toks["input_ids"].squeeze(0), "attention_mask": toks["attention_mask"].squeeze(0)}
        else:
            # Tiny bag-of-words vector
            vec = np.zeros(1024, dtype=np.float32)
            if text:
                for w in text.lower().split():
                    idx = hash(w) % 1024
                    vec[idx] += 1.0
            return {"bow": vec}
