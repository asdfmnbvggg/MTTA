import argparse
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from iopath.common.file_io import g_pathmgr
from prettytable import PrettyTable
from scipy import interpolate
from sklearn import metrics
from PIL import Image

import tent
from utils import get_logger, set_random_seed
from load_Resnet_18 import load_wafer_best_model
from data_unknown import load_wafer

parser = argparse.ArgumentParser()

# Model options
parser.add_argument("--adaptation", default="tent",
                    choices=["source", "tent"])
parser.add_argument("--episodic", action="store_true")
# Optimizer options
parser.add_argument("--steps", default=1, type=int)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--method", default="Adam", choices=["Adam", "SGD"])
parser.add_argument("--momentum", default=0.9, type=float)
# Testing options
parser.add_argument("--batch_size", default=100, type=int)
# Misc options
parser.add_argument("--rng_seed", default=1, type=int)
parser.add_argument("--save_dir", default="./output")
parser.add_argument("--data_dir", default="./data")
parser.add_argument("--ckpt_path", required=True, help="resnet18_wafer_best.pth 경로")
parser.add_argument("--id_pkl", required=True, help="ID test pkl 경로")
parser.add_argument("--ood_pkl", required=True, help="OOD test pkl 경로")
parser.add_argument("--img_col", default="waferMap")
parser.add_argument("--label_col", default="failureType_norm")
parser.add_argument("--normalize", action="store_true")

# CoTTA options
parser.add_argument("--mt", default=0.999, type=float)
parser.add_argument("--rst", default=0.01, type=float)
parser.add_argument("--ap", default=0.92, type=float)
# Tent options
parser.add_argument("--alpha", nargs="+", default=[0.5], type=float)
parser.add_argument("--criterion", default="ent", choices=["ent", "ent_ind", "ent_ind_ood", "ent_unf"])
parser.add_argument("--rounds", default=1, type=int)
# EATA options
parser.add_argument("--fisher_size", default=2000, type=int)
parser.add_argument("--fisher_alpha", default=1., type=float)
parser.add_argument("--e_margin", default=math.log(10)*0.40, type=float)
parser.add_argument("--d_margin", default=0.05, type=float)

args = parser.parse_args()

args.log_dest = f"wafer_{args.adaptation}_lr{args.lr}_steps{args.steps}_alpha{'_'.join(map(str,args.alpha))}_{args.criterion}.txt"

g_pathmgr.mkdirs(args.save_dir)
set_random_seed(args.rng_seed)

logger = get_logger(__name__, args.save_dir, args.log_dest)
logger.info(f"args:\n{args}")

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model, class_to_idx, img_size = load_wafer_best_model(args.ckpt_path, device)

    if args.adaptation == "source":
        base_model.eval()
        model = base_model
    elif args.adaptation == "tent":
        base_model = tent.configure_model(base_model)
        params, _ = tent.collect_params(base_model)
        optimizer = setup_optimizer(params)
        model = tent.Tent(
            base_model, optimizer,
            steps=args.steps,
            episodic=args.episodic,
            alpha=args.alpha,
            criterion=args.criterion
        )
    else:
        raise ValueError(args.adaptation)

    # ✅ 데이터 로드 (한 번만)
    x_ind, y_ind, _ = load_wafer(
        pkl_path=args.id_pkl,
        n_examples=10**9,
        shuffle=False,
        seed=args.rng_seed,
        return_label_mapping=True,
        use_classes=None
    )
    x_ood, _ = load_wafer(
        pkl_path=args.ood_pkl,
        n_examples=10**9,
        shuffle=False,
        seed=args.rng_seed,
        return_label_mapping=False,
        use_classes=None
    )

    x_ind, y_ind, x_ood = x_ind.to(device), y_ind.to(device), x_ood.to(device)

    for r in range(args.rounds):
        acc, (auc, fpr), oscr_ = get_results(model, x_ind, y_ind, x_ood, args.batch_size, device=device)

        t = PrettyTable(["round", "acc", "auroc", "fpr95tpr", "oscr"])
        t.add_row([r, f"{acc:.2%}", f"{auc:.2%}", f"{fpr:.2%}", f"{oscr_:.2%}"])
        logger.info(f"results:\n{t}")


def setup_optimizer(params):
    if args.method == "Adam":
        return optim.Adam(params, lr=args.lr)
    elif args.method == "SGD":
        return optim.SGD(params, args.lr, momentum=args.momentum)
    else:
        raise NotImplementedError


def get_results(model: nn.Module,
                x_ind: torch.Tensor,
                y_ind: torch.Tensor,
                x_ood: torch.Tensor,
                batch_size: int = 100,
                device: torch.device = None):
    if device is None:
        device = x_ind.device

    # ✅ ID/OOD 길이 다를 때 안전하게
    n_total = min(x_ind.shape[0], x_ood.shape[0])
    n_batches = math.ceil(n_total / batch_size)

    acc = 0.0
    y_true = torch.zeros((0,), dtype=torch.float32)
    y_score = torch.zeros((0,), dtype=torch.float32)
    score_ind = torch.zeros((0,), dtype=torch.float32)
    score_ood = torch.zeros((0,), dtype=torch.float32)
    pred = torch.zeros((0,), dtype=torch.long)

    is_tent = (args.adaptation == "tent")

    # tent면 학습 모드/grad 필요, source면 eval/no_grad
    model.train() if is_tent else model.eval()
    context = torch.enable_grad() if is_tent else torch.no_grad()

    with context:
        for counter in range(n_batches):
            s = counter * batch_size
            e = (counter + 1) * batch_size

            x_ind_curr = x_ind[s:e].to(device)
            y_ind_curr = y_ind[s:e].to(device)
            x_ood_curr = x_ood[s:e].to(device)

            # (혹시 마지막 배치에서 한쪽이 비면 스킵)
            if x_ind_curr.shape[0] == 0 or x_ood_curr.shape[0] == 0:
                continue

            x_curr = torch.cat((x_ind_curr, x_ood_curr), dim=0)

            output = model(x_curr)
            energy = output.logsumexp(1)
            prob = output.softmax(1)
            _, pred_ = prob.max(1)

            n_id = x_ind_curr.shape[0]
            acc += (pred_[:n_id] == y_ind_curr).float().sum().item()

            y_true = torch.cat((y_true,
                                torch.cat((torch.ones(n_id), torch.zeros(x_ood_curr.shape[0])))), dim=0)
            y_score = torch.cat((y_score, energy.detach().cpu()), dim=0)

            score_ind = torch.cat((score_ind, energy[:n_id].detach().cpu()), dim=0)
            score_ood = torch.cat((score_ood, energy[n_id:].detach().cpu()), dim=0)

            pred = torch.cat((pred, pred_[:n_id].detach().cpu().long()), dim=0)

    acc = acc / n_total
    auc, fpr = get_ood_metrics(y_true.numpy(), y_score.numpy())
    oscr_ = get_oscr(score_ind.numpy(), score_ood.numpy(), pred.numpy(), y_ind[:n_total].detach().cpu().numpy())
    return acc, (auc, fpr), oscr_


def get_ood_metrics(y_true, y_score):
    auroc = metrics.roc_auc_score(y_true, y_score)
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    return auroc, float(interpolate.interp1d(tpr, fpr)(0.95))


def get_oscr(score_ind, score_ood, pred, y_ind):
    score = np.concatenate((score_ind, score_ood), axis=0)
    def get_fpr(t):
        return (score_ood >= t).sum() / len(score_ood)
    def get_ccr(t):
        return ((score_ind > t) & (pred == y_ind)).sum() / len(score_ind)
    fpr = [0.0]
    ccr = [0.0]
    for s in -np.sort(-score):
        fpr.append(get_fpr(s))
        ccr.append(get_ccr(s))
    fpr.append(1.0)
    ccr.append(1.0)
    roc = sorted(zip(fpr, ccr), reverse=True)
    oscr = 0.0
    for i in range(len(score)):
        oscr += (roc[i][0] - roc[i + 1][0]) * (roc[i][1] + roc[i + 1][1]) / 2.0
    return oscr


if __name__ == "__main__":
    evaluate()
