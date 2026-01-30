from pathlib import Path
from typing import Tuple, Optional, List, Dict, Union

import numpy as np
import pandas as pd
import torch


IMG_COL_CANDIDATES = ["waferMap", "image", "img", "wafer"]
LABEL_COL_CANDIDATES = ["failureType_norm", "failureType", "label", "y"]


def _find_existing_col(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c


def _to_chw_float01(x: np.ndarray) -> np.ndarray:
    x = np.array(x)

    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]

    if x.ndim == 2:
        x = x[None, :, :]

    elif x.ndim == 3 and x.shape[0] != 1:
        x = np.transpose(x, (2, 0, 1))

    x = x.astype(np.float32)

    mx = float(np.max(x)) if x.size > 0 else 0.0
    if mx > 1.5:
        x = x / 255.0

    return x


def load_wafer(
    pkl_path: str,
    n_examples: int,
    shuffle: bool = False,
    seed: int = 0,
    return_label_mapping: bool = False,
    use_classes: Optional[List[str]] = ['Scratch', 'Random', 'Donut', 'Near-full']
) -> Union[
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, Dict[str, int]],
]:

    df = pd.read_pickle(pkl_path)

    img_col = _find_existing_col(df, IMG_COL_CANDIDATES)
    label_col = _find_existing_col(df, LABEL_COL_CANDIDATES)

    df[label_col] = df[label_col].astype(str)

    if use_classes is not None:
        use_set = set(map(str, use_classes))
        df = df[df[label_col].isin(use_set)].copy().reset_index(drop=True)

    y_raw = df[label_col].to_numpy()
    classes = sorted(np.unique(y_raw).tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([class_to_idx[v] for v in y_raw], dtype=np.int64)

    x_list = []
    for im in df[img_col].to_list():
        x_chw = _to_chw_float01(im)
        x_list.append(x_chw)

    x = np.stack(x_list, axis=0)

    N = x.shape[0]
    idx = np.arange(N)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)

    n_take = min(n_examples, N)
    idx = idx[:n_take]
    x = x[idx]
    y = y[idx]

    x_t = torch.tensor(x, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.int64)

    if return_label_mapping:
        return x_t, y_t, class_to_idx

    return x_t, y_t
