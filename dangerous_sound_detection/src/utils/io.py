import numpy as np
from pathlib import Path

def resolve_split_path(path: str, split: str | None = None) -> str:
    """Append the split name to a file path before its suffix."""
    if not split:
        return path
    target = Path(path)
    return str(target.with_name(f"{target.stem}_{split}{target.suffix}"))

def save_features(features: np.ndarray, path: str) -> None:
    """Save features to numpy file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, features)

def load_features(path: str) -> np.ndarray:
    """Load features from numpy file."""
    return np.load(path)

def save_labels(labels: np.ndarray, path: str) -> None:
    """Save labels to numpy file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, labels)

def load_labels(path: str) -> np.ndarray:
    """Load labels from numpy file."""
    return np.load(path)
