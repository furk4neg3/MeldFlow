import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # add project root to path

from mmplat.training.train import main

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config/default.yaml")
    args = ap.parse_args()
    main(args.config)
