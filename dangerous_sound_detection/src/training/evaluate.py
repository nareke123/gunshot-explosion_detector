from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

def evaluate_model(clf, features: np.ndarray, labels: np.ndarray, le) -> dict:
    """Evaluate model."""
    preds_encoded = clf.predict(features)
    preds = le.inverse_transform(preds_encoded)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}
