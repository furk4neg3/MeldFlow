import os
from typing import List, Dict, Any, Optional
import pandas as pd
from torch.utils.data import Dataset

class MultiModalDataset(Dataset):
    """Dataset reading a CSV with columns for image path, text, tabular, and label.
    Missing modalities are allowed; the collate / preprocessors must handle None.
    """
    def __init__(
        self,
        csv_path: str,
        image_column: str,
        text_column: str,
        num_cols: List[str],
        cat_cols: List[str],
        target_column: str,
        image_root: Optional[str] = None,
        drop_all_missing: bool = True,
        split: Optional[str] = None,
        split_column: Optional[str] = None,
    ):
        self.df = pd.read_csv(csv_path)
        if split and split_column and split_column in self.df.columns:
            self.df = self.df[self.df[split_column] == split].reset_index(drop=True)
        self.image_column = image_column
        self.text_column = text_column
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.target_column = target_column
        self.image_root = image_root
        self.drop_all_missing = drop_all_missing

        if drop_all_missing:
            def all_missing(row):
                img = row.get(self.image_column, None)
                txt = row.get(self.text_column, None)
                has_tabular = any([c in row and pd.notna(row[c]) for c in (self.num_cols + self.cat_cols)])
                return (pd.isna(img) or img == "") and (pd.isna(txt) or txt == "") and (not has_tabular)
            self.df = self.df[~self.df.apply(all_missing, axis=1)].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        # Image path resolution
        img_path = row.get(self.image_column, None)
        if isinstance(img_path, str) and self.image_root and not os.path.isabs(img_path):
            img_path = os.path.join(self.image_root, img_path)
        if isinstance(img_path, float):  # NaN
            img_path = None

        # Text
        text = row.get(self.text_column, None)
        if isinstance(text, float):
            text = None

        # Tabular
        num_feats = {c: row[c] for c in self.num_cols if c in row and pd.notna(row[c])}
        cat_feats = {c: row[c] for c in self.cat_cols if c in row and pd.notna(row[c])}
        tabular = {**num_feats, **cat_feats} if (num_feats or cat_feats) else None

        # Target
        y = row.get(self.target_column, None)

        return {
            "image_path": img_path,
            "text": text,
            "tabular": tabular,
            "y": y,
        }
