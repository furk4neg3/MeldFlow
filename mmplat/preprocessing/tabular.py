from typing import Dict, List, Optional
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import joblib
from scipy import sparse  # <-- add this

class TabularPreprocessor:
    def __init__(self, num_cols: List[str], cat_cols: List[str], save_path: Optional[str] = None):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.save_path = save_path
        num_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        cat_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),  # return type may be sparse or dense
        ])
        self.ct = ColumnTransformer(transformers=[
            ("num", num_pipeline, self.num_cols),
            ("cat", cat_pipeline, self.cat_cols),
        ])

    def fit(self, rows: List[Dict]) -> None:
        import pandas as pd
        X = pd.DataFrame([{**{c: None for c in self.num_cols + self.cat_cols}, **(r or {})} for r in rows])
        self.ct.fit(X)

    def transform(self, row: Optional[Dict]) -> np.ndarray:
        import pandas as pd
        X = pd.DataFrame([{**{c: None for c in self.num_cols + self.cat_cols}, **(row or {})}])
        Xt = self.ct.transform(X)
        # Works if Xt is a numpy array OR a scipy sparse matrix
        try:
            Xt = Xt.toarray()
        except AttributeError:
            pass
        Xt = np.asarray(Xt, dtype=np.float32).squeeze(0)
        return Xt

    def save(self):
        if self.save_path:
            joblib.dump(self.ct, self.save_path)

    def load(self):
        if self.save_path:
            self.ct = joblib.load(self.save_path)

    def output_dim(self) -> int:
        dummy = self.transform({})
        return int(dummy.shape[0])
