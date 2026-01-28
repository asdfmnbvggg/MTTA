# train_resnet_wafer_min.py
import os
import random
import argparse
from typing import Dict, Optional

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
    # 기본값(원하면 CLI로 덮어씀)
    pkl_path: str = r"C:\Users\1423\Downloads\MTTA\MTTA-2\MTTA\MTTA\data\LSWD_id_train.pkl"
    test_pkl_path: Optional[str] = None  # ✅ 추가: 외부 테스트 경로
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


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--train_pkl", type=str, default=cfg.pkl_path, help="train pkl path")
    p.add_argument("--test_pkl", type=str, required=True, help="test pkl path")
    p.add_argument("--save_dir", type=str, default=cfg.save_dir)
    p.add_argument("--seed", type=int, default=cfg.seed)

    p.add_argument("--epochs", type=int, default=cfg.epochs)
    p.add_argument("--batch_size", type=int, default=cfg.batch_size)
    p.add_argument("--lr", type=float, default=cfg.lr)
    p.add_argument("--weight_decay", type=float, default=cfg.weight_decay)
    p.add_argument("--num_workers", type=int, default=cfg.num_workers)

    p.add_argument("--img_size", type=int, default=cfg.img_size)
    p.add_argument("--normalize", action="store_true", default=cfg.normalize)
    p.add_argument("--no_normalize", action="store_false", dest="normalize")

    p.add_argument("--use_groupnorm", action="store_true", default=cfg.use_groupnorm)
    p.add_argument("--no_groupnorm", action="store_false", dest="use_groupnorm")
    p.add_argument("--gn_groups", type=int, default=cfg.gn_groups)

    p.add_argument("--test_ratio", type=float, default=cfg.test_ratio, help="used only when test_pkl is None")
    return p


def apply_args_to_cfg(args):
    cfg.pkl_path = args.train_pkl
    cfg.test_pkl_path = args.test_pkl
    cfg.save_dir = args.save_dir
    cfg.seed = args.seed

    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.weight_decay = args.weight_decay
    cfg.num_workers = args.num_workers

    cfg.img_size = args.img_size
    cfg.normalize = args.normalize

    cfg.use_groupnorm = args.use_groupnorm
    cfg.gn_groups = args.gn_groups

    cfg.test_ratio = args.test_ratio


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
    def __init__(self, df, img_col, label_col, class_to_idx: Dict[str, int], transform=None):
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
        1, model.conv1.out_channels,
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
    args = build_parser().parse_args()
    apply_args_to_cfg(args)

    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    df_tr = pd.read_pickle(cfg.pkl_path)
    df_te = pd.read_pickle(cfg.test_pkl_path)

    classes = sorted(df_tr[cfg.label_col].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)

    print(f"[Data] train_pkl={cfg.pkl_path}")
    print(f"[Data] test_pkl={cfg.test_pkl_path}")
    print(f"[Config] num_classes={num_classes}")
    print()

    train_tfms = [T.Resize((cfg.img_size, cfg.img_size)), T.ToTensor()]
    test_tfms  = [T.Resize((cfg.img_size, cfg.img_size)), T.ToTensor()]
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
