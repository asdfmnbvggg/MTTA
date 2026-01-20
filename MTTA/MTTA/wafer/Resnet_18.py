# train_resnet_wafer_min.py
import os
import random
from typing import Dict

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
from torchvision.models import resnet18


class CFG:
    pkl_path: str = r"C:\Users\1423\Downloads\MTTA\MTTA-2\MTTA\MTTA\data\LSWD_id.pkl"
    save_dir: str = r"C:\Users\1423\Downloads\MTTA\-\MTTA\MTTA\wafer"
    seed: int = 1

    img_col: str = "waferMap"
    label_col: str = "failureType_norm"

    test_ratio: float = 0.2
    epochs: int = 30
    batch_size: int = 128

    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 2

    img_size: int = 64
    normalize: bool = True

    use_groupnorm: bool = True
    gn_groups: int = 32


cfg = CFG()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stratified_split_indices(labels: np.ndarray, test_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_ratio))
        test_idx.extend(idx[:n_test].tolist())
        train_idx.extend(idx[n_test:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return np.array(train_idx), np.array(test_idx)


class WaferDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        img_col: str,
        label_col: str,
        class_to_idx: Dict[str, int],
        transform=None,
    ):
        self.df = df.reset_index(drop=True)
        self.img_col = img_col
        self.label_col = label_col
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _to_pil(self, x) -> Image.Image:
        arr = np.array(x)
        arr = np.squeeze(arr)

        if arr.dtype != np.uint8:
            a_min, a_max = arr.min(), arr.max()
            if a_max > a_min:
                arr = (arr - a_min) / (a_max - a_min)
            arr = (arr * 255.0).astype(np.uint8)

        return Image.fromarray(arr, mode="L")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = self._to_pil(row[self.img_col])
        y = self.class_to_idx[str(row[self.label_col])]

        if self.transform is not None:
            img = self.transform(img)
        return img, y


def build_resnet18_1ch(num_classes: int) -> nn.Module:
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(
        1,
        model.conv1.out_channels,
        kernel_size=model.conv1.kernel_size,
        stride=model.conv1.stride,
        padding=model.conv1.padding,
        bias=False,
    )
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def replace_bn_with_gn(module: nn.Module, num_groups: int = 32) -> nn.Module:
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            c = child.num_features
            g = min(num_groups, c)
            while g > 1 and (c % g != 0):
                g -= 1
            setattr(module, name, nn.GroupNorm(num_groups=g, num_channels=c))
        else:
            replace_bn_with_gn(child, num_groups=num_groups)
    return module


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == y).float().mean().item()


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


def main():
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    df = pd.read_pickle(cfg.pkl_path)
    df[cfg.label_col] = df[cfg.label_col].astype(str).str.strip()

    classes = sorted(df[cfg.label_col].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)

    y_all = df[cfg.label_col].map(class_to_idx).to_numpy()

    tr_idx, te_idx = stratified_split_indices(y_all, cfg.test_ratio, cfg.seed)
    df_tr = df.iloc[tr_idx].copy()
    df_te = df.iloc[te_idx].copy()

    print("=== Classes ===")
    print(classes)
    print("\n=== Class counts (train) ===")
    print(df_tr[cfg.label_col].value_counts().to_string())
    print("\n=== Class counts (test) ===")
    print(df_te[cfg.label_col].value_counts().to_string())
    print()

    train_tfms = [T.Resize((cfg.img_size, cfg.img_size)), T.ToTensor()]
    test_tfms = [T.Resize((cfg.img_size, cfg.img_size)), T.ToTensor()]
    if cfg.normalize:
        train_tfms.append(T.Normalize(mean=[0.5], std=[0.5]))
        test_tfms.append(T.Normalize(mean=[0.5], std=[0.5]))

    ds_tr = WaferDataset(df_tr, cfg.img_col, cfg.label_col, class_to_idx, transform=T.Compose(train_tfms))
    ds_te = WaferDataset(df_te, cfg.img_col, cfg.label_col, class_to_idx, transform=T.Compose(test_tfms))

    train_loader = DataLoader(
        ds_tr, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        ds_te, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_resnet18_1ch(num_classes=num_classes)
    if cfg.use_groupnorm:
        model = replace_bn_with_gn(model, num_groups=cfg.gn_groups)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_acc = -1.0
    best_path = os.path.join(cfg.save_dir, "resnet18_wafer_best.pth")

    print(f"[Config] pkl={cfg.pkl_path}")
    print(f"[Config] num_classes={num_classes}")
    print(f"[Config] use_groupnorm={cfg.use_groupnorm}, gn_groups={cfg.gn_groups}, lr={cfg.lr}")

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        te_loss, te_acc = eval_one_epoch(model, test_loader, criterion, device)

        print(f"[Epoch {epoch:03d}] train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"test loss {te_loss:.4f} acc {te_acc:.4f}")

        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_to_idx": class_to_idx,
                    "classes": classes,
                    "img_size": cfg.img_size,
                    "use_groupnorm": cfg.use_groupnorm,
                    "gn_groups": cfg.gn_groups,
                },
                best_path,
            )
            print(f"  -> saved best to: {best_path}")

    print("\nDone.")
    print("Best test acc:", best_acc)
    print("Best ckpt:", best_path)


if __name__ == "__main__":
    main()
