# train_resnet_wafer.py
import os
import random
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import torchvision.transforms as T
from torchvision.models import resnet18


class CFG:
    pkl_path: str = r"C:\Users\1423\Downloads\MTTA\-\MTTA\MTTA\data\LSWMD_prepro.pkl"
    save_dir: str = r"./ckpt_wafer"
    seed: int = 1

    use_classes: Tuple[str, ...] = ("Edge-Ring", "Edge-Loc", "Center", "Loc")

    test_ratio: float = 0.2
    stratified: bool = True

    epochs: int = 20
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 2

    use_weighted_sampler: bool = False   
    use_class_weight_loss: bool = False
    img_size: int = 64
    normalize: bool = True


cfg = CFG()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------
# 2) Load + detect columns
# ----------------------------
def find_column(df: pd.DataFrame, candidates: List[str], kind: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"[{kind}] column not found. Tried: {candidates}\n"
        f"Available columns: {df.columns.tolist()}"
    )


def load_wafer_df(pkl_path: str) -> pd.DataFrame:
    df = pd.read_pickle(pkl_path)
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Loaded object is not a DataFrame: {type(df)}")
    return df


# ----------------------------
# 3) Dataset
# ----------------------------
class WaferDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_col: str, label_col: str,
                 class_to_idx: Dict[str, int], transform=None):
        self.df = df.reset_index(drop=True)
        self.img_col = img_col
        self.label_col = label_col
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _to_pil(self, x) -> Image.Image:
        # x: numpy array or list-like (H,W) or (H,W,1)
        arr = np.array(x)
        arr = np.squeeze(arr)
        if arr.ndim != 2:
            raise ValueError(f"wafer image is not 2D after squeeze. shape={arr.shape}")

        # scale to 0..255 for PIL
        # (이미 uint8이면 그대로)
        if arr.dtype != np.uint8:
            a_min, a_max = arr.min(), arr.max()
            if a_max > a_min:
                arr = (arr - a_min) / (a_max - a_min)
            arr = (arr * 255.0).astype(np.uint8)

        return Image.fromarray(arr, mode="L")  # 1-channel

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = self._to_pil(row[self.img_col])
        y_name = row[self.label_col]
        y = self.class_to_idx[str(y_name)]

        if self.transform is not None:
            img = self.transform(img)
        return img, y


# ----------------------------
# 4) Split (Stratified)
# ----------------------------
def stratified_split_indices(labels: np.ndarray, test_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    classes = np.unique(labels)
    for c in classes:
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_ratio))
        test_idx.extend(idx[:n_test].tolist())
        train_idx.extend(idx[n_test:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return np.array(train_idx), np.array(test_idx)


# ----------------------------
# 5) Imbalance utilities
# ----------------------------
def make_class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    # inverse frequency
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    weights = 1.0 / np.maximum(counts, 1.0)
    weights = weights / weights.sum() * num_classes  # normalize-ish
    return torch.tensor(weights, dtype=torch.float32)


def make_weighted_sampler(y: np.ndarray, num_classes: int) -> WeightedRandomSampler:
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    class_w = 1.0 / np.maximum(counts, 1.0)
    sample_w = class_w[y]
    sample_w = torch.tensor(sample_w, dtype=torch.double)
    return WeightedRandomSampler(weights=sample_w, num_samples=len(sample_w), replacement=True)


# ----------------------------
# 6) Model: ResNet18 for 1-channel
# ----------------------------
def build_resnet18_1ch(num_classes: int) -> nn.Module:
    model = resnet18(weights=None)
    # change first conv from 3ch -> 1ch
    model.conv1 = nn.Conv2d(
        1, model.conv1.out_channels,
        kernel_size=model.conv1.kernel_size,
        stride=model.conv1.stride,
        padding=model.conv1.padding,
        bias=False
    )
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ----------------------------
# 7) Train / Eval loops
# ----------------------------
def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits.detach(), y) * bs
        n += bs
    return total_loss / n, total_acc / n


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits, y) * bs
        n += bs
    return total_loss / n, total_acc / n


# ----------------------------
# 8) Main
# ----------------------------
def main():
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    df = load_wafer_df(cfg.pkl_path)

    # try to detect image/label columns
    img_col = find_column(df, ["waferMap", "wafer_map", "image", "img", "X", "data"], "image")
    label_col = find_column(df, ["failureType_norm", "failureType", "label", "y", "target"], "label")

    # filter classes
    df[label_col] = df[label_col].astype(str)
    df_f = df[df[label_col].isin(cfg.use_classes)].copy()

    if len(df_f) == 0:
        raise ValueError(f"No rows found for classes {cfg.use_classes}. Check label values in '{label_col}'.")

    # map class -> idx (고정 순서)
    class_to_idx = {c: i for i, c in enumerate(cfg.use_classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    num_classes = len(cfg.use_classes)

    # y array for split/sampler
    y_all = df_f[label_col].map(class_to_idx).to_numpy()

    # split
    if cfg.stratified:
        tr_idx, te_idx = stratified_split_indices(y_all, cfg.test_ratio, cfg.seed)
    else:
        rng = np.random.default_rng(cfg.seed)
        perm = rng.permutation(len(df_f))
        n_test = int(round(len(df_f) * cfg.test_ratio))
        te_idx, tr_idx = perm[:n_test], perm[n_test:]

    df_tr = df_f.iloc[tr_idx].copy()
    df_te = df_f.iloc[te_idx].copy()

    # report counts
    print("=== Class counts (total) ===")
    print(df_f[label_col].value_counts().to_string())
    print("\n=== Class counts (train) ===")
    print(df_tr[label_col].value_counts().to_string())
    print("\n=== Class counts (test) ===")
    print(df_te[label_col].value_counts().to_string())
    print()

    # transforms
    train_tfms = [
        T.Resize((cfg.img_size, cfg.img_size)),
        # 불균형 완화에 간접 도움: 약한 증강
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        # wafer는 회전 대칭이 종종 의미 있으니 가볍게
        T.RandomRotation(degrees=10),
        T.ToTensor(),  # [1,H,W], 0..1
    ]
    test_tfms = [
        T.Resize((cfg.img_size, cfg.img_size)),
        T.ToTensor(),
    ]

    if cfg.normalize:
        # 1채널 normalize (대략적인 값; 필요하면 학습셋 통계로 계산 추천)
        train_tfms.append(T.Normalize(mean=[0.5], std=[0.5]))
        test_tfms.append(T.Normalize(mean=[0.5], std=[0.5]))

    train_tfms = T.Compose(train_tfms)
    test_tfms = T.Compose(test_tfms)

    ds_tr = WaferDataset(df_tr, img_col, label_col, class_to_idx, transform=train_tfms)
    ds_te = WaferDataset(df_te, img_col, label_col, class_to_idx, transform=test_tfms)

    # dataloaders with imbalance handling
    y_tr = df_tr[label_col].map(class_to_idx).to_numpy()

    sampler = None
    if cfg.use_weighted_sampler:
        sampler = make_weighted_sampler(y_tr, num_classes)
        train_loader = DataLoader(ds_tr, batch_size=cfg.batch_size, sampler=sampler,
                                  num_workers=cfg.num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True,
                                  num_workers=cfg.num_workers, pin_memory=True)

    test_loader = DataLoader(ds_te, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True)

    # model / loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet18_1ch(num_classes=num_classes).to(device)

    if cfg.use_class_weight_loss:
        class_w = make_class_weights(y_tr, num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_w)
        print("Using class-weighted CE loss:", class_w.detach().cpu().numpy())
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_acc = -1.0
    best_path = os.path.join(cfg.save_dir, "resnet18_wafer_best.pth")

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        te_loss, te_acc = eval_one_epoch(model, test_loader, criterion, device)

        print(f"[Epoch {epoch:03d}] "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"test loss {te_loss:.4f} acc {te_acc:.4f}")

        if te_acc > best_acc:
            best_acc = te_acc
            torch.save({
                "model_state": model.state_dict(),
                "num_classes": num_classes,
                "class_to_idx": class_to_idx,
                "img_size": cfg.img_size,
                "img_col": img_col,
                "label_col": label_col,
            }, best_path)
            print(f"  -> saved best to: {best_path}")

    print("\nDone.")
    print("Best test acc:", best_acc)
    print("Best ckpt:", best_path)
    print("Class mapping:", idx_to_class)


if __name__ == "__main__":
    main()
