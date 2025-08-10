import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from mmplat.preprocessing.tabular import TabularPreprocessor

def test_tabular_preprocessor_fit_transform():
    tp = TabularPreprocessor(num_cols=["n1","n2"], cat_cols=["c1"]
    )
    rows = [{"n1": 1.0, "n2": 2.0, "c1": "A"},
            {"n1": 2.0, "n2": 1.0, "c1": "B"},
            {"n1": None, "n2": 0.0, "c1": None}]
    tp.fit(rows)
    v = tp.transform({"n1": 0.0, "n2": 0.0, "c1": "B"})
    assert v.shape[0] > 0
