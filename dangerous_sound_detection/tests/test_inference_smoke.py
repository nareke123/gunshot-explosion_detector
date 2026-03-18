from pathlib import Path
import uuid
import numpy as np
import soundfile as sf
import joblib
from sklearn.preprocessing import LabelEncoder
from src.inference.predict_audio import predict_audio

class DummyClassifier:
    def __init__(self, predicted_index: int):
        self.predicted_index = predicted_index

    def predict_proba(self, features):
        probs = np.full((len(features), 3), 0.01, dtype=np.float32)
        probs[:, self.predicted_index] = 0.98
        return probs

class DummyExtractor:
    def __init__(self, model_url: str):
        self.model_url = model_url

    def extract(self, audio: np.ndarray, sr: int) -> np.ndarray:
        del audio, sr
        return np.ones((1, 1024), dtype=np.float32)

def test_inference_smoke(monkeypatch):
    monkeypatch.setattr('src.inference.predict_audio.YAMNetExtractor', DummyExtractor)

    tmp_dir = Path('.tmp') / f'inference_{uuid.uuid4().hex}'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    y = np.random.randn(16000).astype(np.float32)
    audio_path = tmp_dir / 'test.wav'
    sf.write(audio_path, y, 16000)

    label_encoder = LabelEncoder()
    label_encoder.fit(['gunshot', 'explosion', 'normal'])
    gunshot_idx = int(np.where(label_encoder.classes_ == 'gunshot')[0][0])
    classifier = DummyClassifier(gunshot_idx)

    model_path = tmp_dir / 'classifier.pkl'
    encoder_path = tmp_dir / 'label_encoder.pkl'
    joblib.dump(classifier, model_path)
    joblib.dump(label_encoder, encoder_path)

    config = {
        'yamnet_model_url': 'unused',
        'model_save_path': str(model_path),
        'label_encoder_path': str(encoder_path),
        'sample_rate': 16000,
        'window_length': 0.96,
        'hop_length': 0.48,
        'confidence_threshold': 0.8,
    }

    result = predict_audio(str(audio_path), config)
    assert 'events' in result
    assert result['events']
    assert result['events'][0]['label'] == 'gunshot'
