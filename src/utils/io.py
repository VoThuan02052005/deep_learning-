import pandas as pd
from pathlib import Path
import pickle

def load_csv(path, parse_dates=None):
    return pd.read_csv(path, parse_dates=parse_dates)

def save_csv(data , path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)

def save_pickle(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)
