from torchvision.models import resnet18
import torch.nn as nn
import torch

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

def load_wafer_best_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    class_to_idx = ckpt.get("class_to_idx", None)
    if class_to_idx is None:
        raise KeyError("ckpt에 'class_to_idx'가 없습니다.")

    num_classes = len(class_to_idx)
    use_groupnorm = bool(ckpt.get("use_groupnorm", True))
    gn_groups = int(ckpt.get("gn_groups", 32))
    img_size = int(ckpt.get("img_size", 64))

    model = build_resnet18_1ch(num_classes=num_classes)
    if use_groupnorm:
        model = replace_bn_with_gn(model, num_groups=gn_groups)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys: {unexpected[:10]}")
    if missing:
        print(f"[Warn] Missing keys: {missing[:10]}")

    model = model.to(device).eval()
    return model, class_to_idx, img_size
