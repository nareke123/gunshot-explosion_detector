import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.features.feature_pipeline import extract_features_for_file
from src.features.yamnet_extractor import YAMNetExtractor
from src.training.torch_model import predict_proba


def _compute_metrics(y_true, y_pred) -> dict:
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average='macro',
        zero_division=0,
    )
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
    }


def evaluate_model(model, features: np.ndarray, labels: np.ndarray, le, device) -> dict:
    """Evaluate model at the window level."""
    preds_encoded = np.argmax(predict_proba(model, features, device), axis=1)
    preds = le.inverse_transform(preds_encoded)
    return _compute_metrics(labels, preds)


def evaluate_file_level(
    model,
    splits_csv: str,
    config: dict,
    le,
    device,
    extractor=None,
) -> dict:
    """Evaluate model at the clip level by averaging probabilities across windows."""
    extractor = extractor or YAMNetExtractor(
        device=config.get('device', 'auto'),
        n_fft=config.get('n_fft', 512),
        hop_length=config.get('stft_hop_length', 160),
        win_length=config.get('win_length', 400),
    )
    df = pd.read_csv(splits_csv)
    y_true = []
    y_pred = []
    for row in df.itertuples(index=False):
        features = extract_features_for_file(
            row.file,
            extractor,
            config['window_length'],
            config['hop_length'],
            config['sample_rate'],
        )
        probs = predict_proba(model, features, device)
        mean_prob = probs.mean(axis=0)
        pred_idx = int(np.argmax(mean_prob))
        y_true.append(row.label)
        y_pred.append(le.inverse_transform([pred_idx])[0])
    return _compute_metrics(y_true, y_pred)
