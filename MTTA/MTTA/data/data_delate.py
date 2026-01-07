import os
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")
"""
Since we will be using labeled data and defective data, 
we exclude and remove that data.
"""

DATA_ROOT = r"C:\Users\1423\Downloads\MTTA\-\MTTA\MTTA\data"
PKL_PATH = os.path.join(DATA_ROOT, "LSWMD.pkl")

df = pd.read_pickle(PKL_PATH)

# Function to normalize labels
def normalize_label(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    if isinstance(x, (list, tuple, np.ndarray)):
        if len(x) == 0:
            return np.nan
        x = x[0]
    return str(x).strip()

tqdm.pandas(desc="Normalizing labels")
df["failureType_norm"] = df["failureType"].progress_apply(normalize_label)

df_labeled = df[df["failureType_norm"].notna()].copy()

label_counts = df_labeled["failureType_norm"].value_counts()

def clean_bracket_string(s):
    if isinstance(s, str):
        s = s.strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1].strip().strip("'").strip('"')
    return s

df_labeled["failureType_norm"] = df_labeled["failureType_norm"].apply(clean_bracket_string)

df_defect = df_labeled[df_labeled["failureType_norm"] != "none"].copy()

removed = len(df_labeled) - len(df_defect)

defect_counts = df_defect["failureType_norm"].value_counts()
print(defect_counts.to_string(), flush=True)

defect_pkl = os.path.join(DATA_ROOT, "LSWMD_defect.pkl")

df_defect.to_pickle(defect_pkl)
