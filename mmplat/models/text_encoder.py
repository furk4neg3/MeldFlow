import torch
import torch.nn as nn

class TextEncoderHF(nn.Module):
    def __init__(self, model_name: str = "distilbert-base-uncased", out_dim: int = 128):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.proj = nn.Linear(hidden_size, out_dim)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # CLS token (DistilBERT uses first token as pooled representation approach)
        cls = out.last_hidden_state[:,0,:]
        return self.proj(cls)

class TextEncoderBOW(nn.Module):
    def __init__(self, in_dim: int = 1024, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )
    def forward(self, bow):
        return self.net(bow)
