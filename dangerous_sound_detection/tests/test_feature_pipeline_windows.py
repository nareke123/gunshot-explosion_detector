from pathlib import Path
import uuid

import numpy as np
import soundfile as sf

from src.features.acoustic_features import ACOUSTIC_FEATURE_NAMES
from src.features.feature_pipeline import extract_all_features, extract_features_for_file
from src.utils.io import load_features, load_labels


class DummyExtractor:
    feature_dim = 4

    def extract(self, audio: np.ndarray, sr: int) -> np.ndarray:
        del sr
        mean_value = float(np.mean(audio)) if audio.size else 0.0
        return np.full((3, self.feature_dim), mean_value, dtype=np.float32)


def test_extract_features_for_file_returns_one_vector_per_window():
    tmp_dir = Path(".tmp") / f"feature_windows_{uuid.uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    audio_path = tmp_dir / "sample.wav"
    y = np.linspace(-1.0, 1.0, 16000 * 2, dtype=np.float32)
    sf.write(audio_path, y, 16000)

    features = extract_features_for_file(
        str(audio_path),
        DummyExtractor(),
        window_length=0.96,
        hop_length=0.48,
        sr=16000,
    )

    assert features.ndim == 2
    assert features.shape[0] == 4
    assert features.shape[1] == 4 + len(ACOUSTIC_FEATURE_NAMES)


def test_extract_all_features_repeats_labels_for_each_window():
    tmp_dir = Path(".tmp") / f"feature_split_{uuid.uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    audio_path = tmp_dir / "sample.wav"
    csv_path = tmp_dir / "split.csv"
    features_path = tmp_dir / "features.npy"
    labels_path = tmp_dir / "labels.npy"

    y = np.linspace(-1.0, 1.0, 16000 * 2, dtype=np.float32)
    sf.write(audio_path, y, 16000)
    csv_path.write_text(f"file,label\n{audio_path.as_posix()},sharp_impulse\n", encoding="utf-8")

    extract_all_features(
        str(csv_path),
        DummyExtractor(),
        {
            "window_length": 0.96,
            "hop_length": 0.48,
            "sample_rate": 16000,
            "features_path": str(features_path),
            "labels_path": str(labels_path),
        },
    )

    features = load_features(str(features_path))
    labels = load_labels(str(labels_path))

    assert features.shape == (4, 4 + len(ACOUSTIC_FEATURE_NAMES))
    assert labels.shape == (4,)
    assert set(labels.tolist()) == {"sharp_impulse"}
