import os, sys, torch
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from mmplat.models.image_encoder import ImageEncoder
from mmplat.models.tabular_encoder import TabularEncoder
from mmplat.models.text_encoder import TextEncoderBOW
from mmplat.models.fusion import MultiModalModel, PredictionHead

def test_model_forward_shapes():
    B = 2
    img_enc = ImageEncoder(name="resnet18", pretrained=False, out_dim=64)
    tab_enc = TabularEncoder(in_dim=10, hidden_dims=[8], out_dim=16)
    txt_enc = TextEncoderBOW(in_dim=1024, out_dim=32)
    head = PredictionHead(in_dim=64+16+32, hidden_dims=[16], out_dim=3, task_type="classification")
    model = MultiModalModel(img_enc, tab_enc, txt_enc, fusion="concat", head=head, task_type="classification")

    batch = {
        "image_tensor": torch.zeros(B,3,224,224),
        "tabular_tensor": torch.zeros(B,10),
        "bow": torch.zeros(B,1024),
    }
    logits = model(batch)
    assert logits.shape == (B,3)
