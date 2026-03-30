from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.features.yamnet_extractor import resolve_device


class AudioClassifier(nn.Module):
    """Simple MLP classifier over aggregated audio features."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def save_torch_classifier(
    path: str,
    model: AudioClassifier,
    input_dim: int,
    hidden_dim: int,
    num_classes: int,
    dropout: float,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_classes": num_classes,
            "dropout": dropout,
        },
        target,
    )


def load_torch_classifier(path: str, device: str = "auto") -> tuple[AudioClassifier, torch.device]:
    resolved_device = resolve_device(device)
    checkpoint = torch.load(path, map_location=resolved_device)
    model = AudioClassifier(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_classes=checkpoint["num_classes"],
        dropout=checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(resolved_device)
    model.eval()
    return model, resolved_device


def predict_proba(model: AudioClassifier, features, device: torch.device) -> torch.Tensor:
    with torch.inference_mode():
        feature_array = np.asarray(features, dtype=np.float32)
        feature_tensor = torch.as_tensor(feature_array, dtype=torch.float32, device=device)
        if feature_tensor.ndim == 1:
            feature_tensor = feature_tensor.unsqueeze(0)
        logits = model(feature_tensor)
        return torch.softmax(logits, dim=-1).detach().cpu().numpy()
