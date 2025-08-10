import base64, json, argparse, requests

def b64_image(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://127.0.0.1:8000")
    ap.add_argument("--image", default=None)
    ap.add_argument("--text", default="a red square high value")
    ap.add_argument("--tabular_json", default='{"num_a": 0.2, "num_b": 1.1, "cat_x": "A"}')
    args = ap.parse_args()

    payload = {
        "text": args.text,
        "tabular": json.loads(args.tabular_json),
        "image_b64": b64_image(args.image) if args.image else None
    }
    r = requests.post(args.host + "/predict", json=payload, timeout=30)
    print(r.status_code, r.json())
