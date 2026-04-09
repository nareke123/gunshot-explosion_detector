from pathlib import Path
import uuid
import numpy as np
import soundfile as sf
import torch
from torch import nn
from sklearn.preprocessing import LabelEncoder
from src.inference.predict_audio import predict_audio


class DummyClassifier(nn.Module):
    def __init__(self, predicted_index: int, num_classes: int):
        super().__init__()
        self.predicted_index = predicted_index
        self.num_classes = num_classes

    def forward(self, features):
        logits = torch.full((features.shape[0], self.num_classes), -4.0, dtype=torch.float32, device=features.device)
        logits[:, self.predicted_index] = 4.0
        return logits

class DummyExtractor:
    def extract(self, audio: np.ndarray, sr: int) -> np.ndarray:
        del audio, sr
        return np.ones((1, 257), dtype=np.float32)

def test_inference_smoke():
    tmp_dir = Path('.tmp') / f'inference_{uuid.uuid4().hex}'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    y = np.random.randn(16000).astype(np.float32)
    audio_path = tmp_dir / 'test.wav'
    sf.write(audio_path, y, 16000)

    label_encoder = LabelEncoder()
    label_encoder.fit(['danger_noise', 'normal'])
    event_idx = int(np.where(label_encoder.classes_ == 'danger_noise')[0][0])
    classifier = DummyClassifier(event_idx, len(label_encoder.classes_))

    config = {
        'sample_rate': 16000,
        'window_length': 0.96,
        'hop_length': 0.48,
        'confidence_threshold': 0.8,
        'background_labels': ['normal'],
        'event_labels': ['safety_noise', 'discipline_violation', 'danger_noise'],
        'alert_labels': ['danger_noise'],
        'model_class_names': ['normal', 'safety_noise', 'discipline_violation', 'danger_noise'],
        'event_gate_min_rms_dbfs': -120.0,
        'event_gate_min_peak_dbfs': -120.0,
        'event_gate_min_rms_above_noise_floor_db': 0.0,
        'event_gate_min_crest_factor': 1.0,
    }

    result = predict_audio(
        str(audio_path),
        config,
        extractor=DummyExtractor(),
        model=classifier,
        le=label_encoder,
        device='cpu',
        include_windows=True,
    )
    assert 'events' in result
    assert result['events']
    assert result['events'][0]['label'] == 'danger_noise'
    assert 'activities' in result
    assert result['activities']
    assert 'windows' in result
    assert result['windows']
    assert result['windows'][0]['predicted_label'] == 'danger_noise'
    assert result['windows'][0]['event_gate']['passed'] is True
