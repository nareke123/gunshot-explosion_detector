from sklearn.preprocessing import LabelEncoder
import joblib
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.features.yamnet_extractor import resolve_device
from src.training.torch_model import AudioClassifier, predict_proba, save_torch_classifier

def train_classifier(features: np.ndarray, labels: np.ndarray, config: dict) -> None:
    """Train a PyTorch classifier on extracted audio-window features."""
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    model_dir = Path(config['model_save_path']).parent
    model_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.get('device', 'auto'))
    input_dim = features.shape[1]
    hidden_dim = int(config.get('hidden_dim', 256))
    dropout = float(config.get('dropout', 0.2))
    num_epochs = int(config.get('num_epochs', 20))
    batch_size = int(config.get('batch_size', 64))
    learning_rate = float(config.get('learning_rate', 1e-3))
    use_class_weights = bool(config.get('use_class_weights', True))

    model = AudioClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=len(le.classes_),
        dropout=dropout,
    ).to(device)

    dataset = TensorDataset(
        torch.as_tensor(features, dtype=torch.float32),
        torch.as_tensor(labels_encoded, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    if use_class_weights:
        class_counts = np.bincount(labels_encoded, minlength=len(le.classes_)).astype(np.float32)
        class_weights = class_counts.sum() / (len(le.classes_) * np.maximum(class_counts, 1.0))
        weight_tensor = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    else:
        class_weights = None
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = []
    model.train()
    for _ in range(num_epochs):
        epoch_loss = 0.0
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            logits = model(batch_features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_features.size(0)
        history.append(epoch_loss / len(dataset))

    save_torch_classifier(
        config['model_save_path'],
        model,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=len(le.classes_),
        dropout=dropout,
    )
    joblib.dump(le, config['label_encoder_path'])

    # Save metrics (on train for demo)
    probs = predict_proba(model, features, device)
    preds = np.argmax(probs, axis=1)
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
    report['device'] = str(device)
    report['train_loss'] = history[-1] if history else None
    report['class_weights'] = (
        {str(class_name): float(weight) for class_name, weight in zip(le.classes_, class_weights)}
        if class_weights is not None
        else None
    )
    with open(model_dir / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(report, f)
