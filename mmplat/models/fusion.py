import torch
import torch.nn as nn

class ConcatFusion(nn.Module):
    def forward(self, embs):
        return torch.cat(embs, dim=-1)

def mlp(in_dim: int, hidden_dims, out_dim: int, dropout: float = 0.0):
    dims = [in_dim] + list(hidden_dims) + [out_dim]
    layers = []
    for i in range(len(dims)-2):
        layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU()]
        if dropout > 0:
            layers += [nn.Dropout(dropout)]
    layers += [nn.Linear(dims[-2], dims[-1])]
    return nn.Sequential(*layers)

class PredictionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dims=(128,), out_dim: int = 3, task_type: str = "classification", dropout: float = 0.1):
        super().__init__()
        self.task_type = task_type
        self.mlp = mlp(in_dim, hidden_dims, out_dim, dropout)

    def forward(self, x):
        return self.mlp(x)

class MultiModalModel(nn.Module):
    def __init__(self, image_encoder=None, tabular_encoder=None, text_encoder=None, fusion="concat", head=None, task_type="classification"):
        super().__init__()
        self.image_encoder = image_encoder
        self.tabular_encoder = tabular_encoder
        self.text_encoder = text_encoder
        self.task_type = task_type

        # Normalize fusion config (string or dict) and default to concat
        fusion_type = fusion.get("type", "concat") if isinstance(fusion, dict) else fusion
        if fusion_type is None or fusion_type == "concat":
            self.fusion = ConcatFusion()
        else:
            # Only concat is implemented; default to concat to be resilient
            self.fusion = ConcatFusion()

        self.head = head

    def forward(self, batch):
        embs = []
        if self.image_encoder is not None and batch.get("image_tensor") is not None:
            img = batch.get("image_tensor")
            embs.append(self.image_encoder(img))
        if self.tabular_encoder is not None and batch.get("tabular_tensor") is not None:
            tab = batch.get("tabular_tensor")
            embs.append(self.tabular_encoder(tab))
        if self.text_encoder is not None:
            if "input_ids" in batch:
                txt = self.text_encoder(batch["input_ids"], batch["attention_mask"])
            elif "bow" in batch:
                txt = self.text_encoder(batch["bow"])
            else:
                txt = None
            if txt is not None:
                embs.append(txt)

        if not embs:
            raise ValueError("No modality available in batch.")

        if len(embs) > 1:
            # Use fusion module if present; otherwise safe concat fallback
            joint = self.fusion(embs) if hasattr(self, "fusion") and self.fusion is not None else torch.cat(embs, dim=-1)
        else:
            joint = embs[0]

        logits = self.head(joint)
        return logits
