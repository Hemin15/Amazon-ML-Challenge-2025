# utils.py
import os
import random
import numpy as np
import torch
import pandas as pd

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass

def safe_read_csv(path, nrows=None):
    """
    Robust CSV read: try UTF-8 then ISO-8859-1, skip bad lines.
    """
    kwargs = dict(sep=',', engine='python', quoting=1, on_bad_lines='skip', keep_default_na=True)
    if nrows:
        kwargs['nrows'] = nrows
    try:
        return pd.read_csv(path, encoding='utf-8', **kwargs)
    except Exception:
        return pd.read_csv(path, encoding='ISO-8859-1', **kwargs)

def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)
