from typing import Optional
from PIL import Image
import torch
from torchvision import transforms

class ImagePreprocessor:
    def __init__(self, size: int = 224, mean=None, std=None, augment: bool = True):
        mean = mean or [0.485, 0.456, 0.406]
        std = std or [0.229, 0.224, 0.225]
        if augment:
            tr = [
                transforms.Resize((size, size)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.1,0.1,0.1,0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        else:
            tr = [
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        self.transform = transforms.Compose(tr)
        self.size = size
        self.zero_image = torch.zeros(3, size, size)

    def __call__(self, image_path: Optional[str]) -> torch.Tensor:
        if image_path is None or image_path == "":
            return self.zero_image
        try:
            img = Image.open(image_path).convert("RGB")
            return self.transform(img)
        except Exception:
            return self.zero_image
