from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from pathlib import Path

def train_classifier(features: np.ndarray, labels: np.ndarray, config: dict) -> None:
    """Train classifier on features."""
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    if config['classifier_type'] == 'logistic_regression':
        clf = LogisticRegression(random_state=42, max_iter=1000)
    elif config['classifier_type'] == 'xgboost':
        from xgboost import XGBClassifier
        clf = XGBClassifier(random_state=42, eval_metric='mlogloss')
    else:
        raise ValueError(f"Unsupported classifier_type: {config['classifier_type']}")

    model_dir = Path(config['model_save_path']).parent
    model_dir.mkdir(parents=True, exist_ok=True)

    clf.fit(features, labels_encoded)
    joblib.dump(clf, config['model_save_path'])
    joblib.dump(le, config['label_encoder_path'])

    # Save metrics (on train for demo)
    preds = clf.predict(features)
    cm = confusion_matrix(labels_encoded, preds)
    plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(model_dir / 'confusion_matrix.png')
    plt.close()

    report = classification_report(
        labels_encoded,
        preds,
        target_names=[str(class_name) for class_name in le.classes_],
        output_dict=True,
        zero_division=0,
    )
    with open(model_dir / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(report, f)
