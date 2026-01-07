import os
import numpy as np
import pandas as pd
from PIL import Image


DATA_ROOT = r"C:\Users\1423\Downloads\MTTA\-\MTTA\MTTA\data"
DEFECT_PKL = os.path.join(DATA_ROOT, "LSWMD_defect.pkl")

OUT_PKL = os.path.join(DATA_ROOT, "LSWMD_prepro.pkl")

df = pd.read_pickle(DEFECT_PKL)

print("df shape:", df.shape)
print("df columns:", df.columns.tolist())

def to_2d_wafer(arr) -> np.ndarray:
    x = np.array(arr)
    x = np.squeeze(x)
    if x.ndim != 2:
        raise ValueError(f"waferMap is not 2D. shape={x.shape}")
    return x


def resize_nearest(x: np.ndarray, target_hw=(64, 64)) -> np.ndarray:
    th, tw = target_hw
    im = Image.fromarray(x.astype(np.uint8), mode="L")
    im = im.resize((tw, th), resample=Image.NEAREST)
    return np.array(im)

LABEL_COL = "failureType_norm"
keep_cols = ["waferMap", LABEL_COL]

missing = [c for c in keep_cols if c not in df.columns]
if missing:
    raise KeyError(f"Missing columns: {missing}")

df = df[keep_cols].copy()
print("Kept columns:", df.columns.tolist())

shapes = []
for wm in df["waferMap"].values:
    shapes.append(to_2d_wafer(wm).shape)

shapes_series = pd.Series(shapes).value_counts()
print("Top-10 shapes:")
print(shapes_series.head(10).to_string())

max_h = max(s[0] for s in shapes)
max_w = max(s[1] for s in shapes)
print("Max shape:", (max_h, max_w))

labels = df[LABEL_COL].astype(str).values
classes = sorted(pd.unique(labels))
class_to_idx = {c: i for i, c in enumerate(classes)}
y = np.array([class_to_idx[c] for c in labels], dtype=np.int64)

print("Classes:", classes)
print("Label counts:")
print(pd.Series(labels).value_counts().to_string())


TARGET_SIZE = (64, 64)

wafer64_list = []
for wm in df["waferMap"].values:
    x = to_2d_wafer(wm)
    out = resize_nearest(x, TARGET_SIZE) 
    wafer64_list.append(out.astype(np.uint8))

df["waferMap"] = wafer64_list

df.to_pickle(OUT_PKL)
print("Saved:", OUT_PKL)
