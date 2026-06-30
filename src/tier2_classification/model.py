import torch
import torch.nn as nn
from pathlib import Path

try:
    from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
except Exception:
    from torchvision.models import efficientnet_b4
    EfficientNet_B4_Weights = None

def build_model(num_classes: int, pretrained: bool = True, freeze_backbone: bool = False, weights_path: str = None, device: str = "cpu"):
    if EfficientNet_B4_Weights is not None and pretrained:
        weights = EfficientNet_B4_Weights.DEFAULT
        model = efficientnet_b4(weights=weights)
    else:
        model = efficientnet_b4(weights=None)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        for name, param in model.features.named_parameters():
            param.requires_grad = False

    if weights_path and Path(weights_path).exists():
        state = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state, strict=False)

    model.to(device)
    return model

def unfreeze_all(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = True
    return model