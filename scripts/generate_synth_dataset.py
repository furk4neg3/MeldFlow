import os, json, random, argparse
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont

RNG = random.Random(1337)
NP_RNG = np.random.default_rng(1337)

CLASSES = [0,1,2]
COLORS = [(220,30,30), (30,180,50), (40,80,220)]  # red, green, blue

TEMPLATES = {
    0: ["red square low value", "crimson box small", "scarlet square tiny"],
    1: ["green circle medium value", "emerald disk medium", "greenish round"],
    2: ["blue triangle high value", "azure tri large", "blue triangular big"],
}

def draw_shape(img_size, cls):
    img = Image.new("RGB", (img_size, img_size), (255,255,255))
    draw = ImageDraw.Draw(img)
    color = COLORS[cls]
    pad = img_size // 8
    if cls == 0:
        # square
        draw.rectangle([pad, pad, img_size-pad, img_size-pad], outline=color, fill=color)
    elif cls == 1:
        # circle
        draw.ellipse([pad, pad, img_size-pad, img_size-pad], outline=color, fill=color)
    else:
        # triangle
        draw.polygon([(img_size//2, pad), (img_size-pad, img_size-pad), (pad, img_size-pad)], outline=color, fill=color)
    return img

def main(out_dir, num_samples=600, image_size=64):
    out = Path(out_dir)
    (out/"images").mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(num_samples):
        cls = RNG.choice(CLASSES)
        # Image
        img = draw_shape(image_size, cls)
        img_path = f"images/sample_{i}.png"
        img.save(out/img_path)

        # Tabular (num_a, num_b correlated with class; cat_x A/B/C)
        base = float(cls)  # 0,1,2
        num_a = base + NP_RNG.normal(0, 0.2)
        num_b = (2-base) + NP_RNG.normal(0, 0.2)
        cat_x = ["A","B","C"][cls]

        # Text
        text = RNG.choice(TEMPLATES[cls])

        # Split
        if i < int(0.7*num_samples):
            split = "train"
        elif i < int(0.85*num_samples):
            split = "val"
        else:
            split = "test"

        rows.append({
            "image_path": f"sample_{i}.png",
            "text": text,
            "num_a": round(float(num_a), 4),
            "num_b": round(float(num_b), 4),
            "cat_x": cat_x,
            "label": cls,
            "split": split,
        })
    df = pd.DataFrame(rows)
    df.to_csv(out/"synth.csv", index=False)
    print(f"Wrote {len(rows)} rows to {out/'synth.csv'} and images to {out/'images'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/synth")
    ap.add_argument("--num_samples", type=int, default=600)
    ap.add_argument("--image_size", type=int, default=64)
    args = ap.parse_args()
    main(args.out_dir, args.num_samples, args.image_size)
