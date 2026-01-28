# train_resnet_wafer_argparse.py
import os
import random
from typing import Dict, List

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
from torchvision.models import resnet18

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--train_pkl", required=True, help="Train pkl path")
parser.add_argument("--test_pkl", required=True, help="Test pkl path")
parser.add_argument("--save_dir", required=True, help="Checkpoint save dir")

parser.add_argument("--img_col", default="waferMap")
parser.add_argument("--label_col", default="failureType_norm")

parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--num_workers", type=int, default=2)

parser.add_argument("--img_size", type=int, default=64)
parser.add_argument("--normalize", action="store_true")

parser.add_argument("--seed", type=int, default=1)

args = parser.parse_args()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class WaferDataset(Dataset):
    def __init__(self, df, img_col, label_col, class_to_idx, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_col = img_col
        self.label_col = label_col
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _to_pil_1ch(x):
        arr = np.squeeze(np.array(x))
        if arr.dtype != np.uint8:
            a_min, a_max = arr.min(), arr.max()
            if a_max > a_min:
                arr = (arr - a_min) / (a_max - a_min)
            arr = (arr * 255).astype(np.uint8)
        return Image.fromarray(arr, mode="L")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = self._to_pil_1ch(row[self.img_col])
        y = self.class_to_idx[str(row[self.label_col]).strip()]
        if self.transform:
            img = self.transform(img)
        return img, y


def build_resnet18_1ch(num_classes):
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


def accuracy(logits, y):
    return (logits.argmax(1) == y).float().mean().item()


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    loss_sum, acc_sum, n = 0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        loss_sum += loss.item() * bs
        acc_sum += accuracy(logits.detach(), y) * bs
        n += bs

    return loss_sum / n, acc_sum / n


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    loss_sum, acc_sum, n = 0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        bs = x.size(0)
        loss_sum += loss.item() * bs
        acc_sum += accuracy(logits, y) * bs
        n += bs

    return loss_sum / n, acc_sum / n

def main():
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    df_tr = pd.read_pickle(args.train_pkl)
    df_te = pd.read_pickle(args.test_pkl)

    df_tr[args.label_col] = df_tr[args.label_col].astype(str).str.strip()
    df_te[args.label_col] = df_te[args.label_col].astype(str).str.strip()

    classes = sorted(df_tr[args.label_col].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)

    unknown = set(df_te[args.label_col].unique()) - set(classes)
    if unknown:
        raise ValueError(f"Test set has unknown labels: {unknown}")

    print("Classes:", classes)
    print("\nTrain counts:\n", df_tr[args.label_col].value_counts())
    print("\nTest counts:\n", df_te[args.label_col].value_counts())

    tfms = [T.Resize((args.img_size, args.img_size)), T.ToTensor()]
    if args.normalize:
        tfms.append(T.Normalize([0.5], [0.5]))
    tfms = T.Compose(tfms)

    ds_tr = WaferDataset(df_tr, args.img_col, args.label_col, class_to_idx, tfms)
    ds_te = WaferDataset(df_te, args.img_col, args.label_col, class_to_idx, tfms)

    train_loader = DataLoader(
        ds_tr, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        ds_te, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_resnet18_1ch(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_acc = -1.0
    best_path = os.path.join(args.save_dir, "resnet18_wafer_best.pth")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        te_loss, te_acc = eval_one_epoch(model, test_loader, criterion, device)

        print(f"[Epoch {epoch:03d}] "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"test loss {te_loss:.4f} acc {te_acc:.4f}")

        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_to_idx": class_to_idx,
                    "classes": classes,
                    "img_size": args.img_size,
                },
                best_path,
            )
            print("Saved best:", best_path)

    print("\nBest acc:", best_acc)


if __name__ == "__main__":
    main()
