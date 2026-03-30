from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

from src.training.torch_model import predict_proba

def evaluate_model(model, features: np.ndarray, labels: np.ndarray, le, device) -> dict:
    """Evaluate model."""
    preds_encoded = np.argmax(predict_proba(model, features, device), axis=1)
    preds = le.inverse_transform(preds_encoded)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}
