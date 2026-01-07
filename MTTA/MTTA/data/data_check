import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_ROOT = r"C:\Users\1423\Downloads\MTTA\-\MTTA\MTTA\data"
PKL_64 = os.path.join(DATA_ROOT, "LSWMD_prepro.pkl")

LABEL_COL = "failureType_norm"
IMG_COL = "waferMap"
SEED = 42

def inspect_wafer_pkl(
    pkl_path: str,
    img_col: str = "waferMap",
    label_col: str = "failureType_norm",
    n_shape_check: int = 10,
):
    df = pd.read_pickle(pkl_path)

    print("=" * 80)
    print("Loaded:", pkl_path)
    print("df shape:", df.shape)
    print("df columns:", df.columns.tolist())
    print("\nDtypes:")
    print(df.dtypes)

    if img_col not in df.columns:
        raise KeyError(f"Missing image column: {img_col}")
    if label_col not in df.columns:
        raise KeyError(f"Missing label column: {label_col}")

    labels = df[label_col].astype(str)
    classes = sorted(labels.unique().tolist())
    print("\nClasses:", classes)
    print("\nLabel counts:")
    print(labels.value_counts().to_string())

    print("\nSample waferMap shape checks:")
    for i in range(min(n_shape_check, len(df))):
        arr = np.array(df[img_col].iloc[i])
        print(f"  idx={i} shape={arr.shape} dtype={arr.dtype}")

    sample_idx = np.random.RandomState(SEED).choice(len(df), size=min(1000, len(df)), replace=False)
    ok = True
    for i in sample_idx:
        arr = np.array(df[img_col].iloc[i])
        if arr.shape != (64, 64):
            ok = False
            print(f"\n[WARN] Found non-64x64 at idx={i}: shape={arr.shape}")
            break
    if ok:
        print("\n[OK] (sampled) All checked waferMap shapes are (64, 64).")

    print("=" * 80)
    return df


def show_images_per_class(
    df: pd.DataFrame,
    img_col: str = "waferMap",
    label_col: str = "failureType_norm",
    n_per_class: int = 3,
    random: bool = True,
    seed: int = 42,
):
    rng = np.random.RandomState(seed)
    classes = sorted(df[label_col].astype(str).unique().tolist())

    class_to_indices = {
        c: df.index[df[label_col].astype(str) == c].to_numpy()
        for c in classes
    }

    n_rows = len(classes)
    n_cols = n_per_class
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.8 * n_rows))

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([[ax] for ax in axes])

    for r, c in enumerate(classes):
        idxs = class_to_indices[c]
        if len(idxs) == 0:
            for cc in range(n_cols):
                axes[r, cc].axis("off")
            continue

        if random:
            pick = rng.choice(idxs, size=min(n_cols, len(idxs)), replace=False)
        else:
            pick = idxs[:min(n_cols, len(idxs))]

        pick = list(pick) + [None] * (n_cols - len(pick))

        for cc, idx in enumerate(pick):
            ax = axes[r, cc]
            ax.set_xticks([])
            ax.set_yticks([])

            if idx is None:
                ax.axis("off")
                continue

            img = np.array(df.loc[idx, img_col])
            ax.imshow(img, cmap="gray")
            ax.set_title(f"{c} | idx={idx}", fontsize=9)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df64 = inspect_wafer_pkl(PKL_64, img_col=IMG_COL, label_col=LABEL_COL)
    show_images_per_class(df64, img_col=IMG_COL, label_col=LABEL_COL, n_per_class=3, random=True, seed=SEED)
